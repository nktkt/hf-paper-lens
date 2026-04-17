"""
Runnable demo for Switch-KD.

- Teacher VLM: larger hidden dim, pretrained on synthetic "image →
  caption id" data so it is a useful teacher (not random noise).
- Student VLM: smaller hidden dim, trained purely via Switch-KD.
- Synthetic data: image ∈ R^{image_dim} (flat features), caption ∈
  sequence of token ids sampled from a fixed palette conditioned on the
  image bucket.

This demonstrates that the Switch-KD loss *actually decreases*, the
student vocabulary agreement improves, and the visual-switch path
trains in the presence of a frozen teacher.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from switch_kd import SwitchKD


# ---------------------------------------------------------------------------
# Tiny VLM used for both teacher and student
# ---------------------------------------------------------------------------


@dataclass
class VLMConfig:
    image_dim: int = 64
    num_visual_tokens: int = 4
    vocab_size: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_text_len: int = 16


class DummyVLM(nn.Module):
    """Vision encoder + projector + causal text decoder over concatenated
    [visual_tokens ; text_ids] sequence.  Text logits are returned."""

    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_enc = nn.Sequential(
            nn.Linear(cfg.image_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.num_visual_tokens * cfg.d_model),
        )
        self.projector = nn.Linear(cfg.d_model, cfg.d_model)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(
            cfg.num_visual_tokens + cfg.max_text_len, cfg.d_model
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def vision(self, image: torch.Tensor) -> torch.Tensor:
        B = image.size(0)
        v = self.vision_enc(image)
        return v.view(B, self.cfg.num_visual_tokens, self.cfg.d_model)

    def project(self, v: torch.Tensor) -> torch.Tensor:
        return self.projector(v)

    def language(
        self, visual_tokens: torch.Tensor, text_ids: torch.Tensor
    ) -> torch.Tensor:
        B, V_tok, D = visual_tokens.shape
        T = text_ids.size(1)
        txt = self.tok_emb(text_ids)
        x = torch.cat([visual_tokens, txt], dim=1)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(
            x.size(1), device=x.device
        )
        h = self.decoder(x, mask=mask, is_causal=True)
        text_h = h[:, V_tok:]
        return self.lm_head(text_h)


# ---------------------------------------------------------------------------
# Synthetic image → caption-id data
# ---------------------------------------------------------------------------


def make_synth_batch(
    batch_size: int, cfg: VLMConfig, seq_len: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each image has a latent bucket; the caption is a *deterministic*
    token sequence determined by that bucket (so teacher can achieve
    high accuracy and argmax-agreement is a meaningful metric)."""
    num_buckets = 8
    bucket = torch.randint(0, num_buckets, (batch_size,), device=device)
    base = 0.2 * torch.randn(batch_size, cfg.image_dim, device=device)
    image = base + bucket.unsqueeze(-1).float() * 0.8

    # deterministic caption: 16 consecutive token ids starting at bucket*8
    start = (bucket * 8).clamp(max=cfg.vocab_size - seq_len)
    text_ids = start.unsqueeze(-1) + torch.arange(seq_len, device=device)
    return image, text_ids


def pretrain_teacher(
    teacher: DummyVLM, cfg: VLMConfig, device: str, steps: int = 300
) -> None:
    opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    teacher.train()
    for _ in range(steps):
        image, text_ids = make_synth_batch(32, cfg, cfg.max_text_len, device)
        v = teacher.project(teacher.vision(image))
        logits = teacher.language(v, text_ids[:, :-1])
        target = text_ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size), target.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main(steps: int = 300, device: str = "cpu", seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    # Switch-KD in the paper uses TinyLLaVA (0.5B) ↔ 3B teacher where the
    # language backbone shares hidden dim; the student only differs in
    # depth.  We mirror that here so the visual-switch path is
    # dimension-compatible without a hand-tuned adapter.
    teacher_cfg = VLMConfig(d_model=64, n_layers=4)
    student_cfg = VLMConfig(d_model=64, n_layers=2)

    teacher = DummyVLM(teacher_cfg).to(device)
    student = DummyVLM(student_cfg).to(device)

    # Teacher and student must share vocab for DBiLD to be comparable
    assert teacher_cfg.vocab_size == student_cfg.vocab_size

    print("[1/3] pretraining teacher on synthetic captioning …")
    pretrain_teacher(teacher, teacher_cfg, device, steps=400)

    # Reality check: teacher accuracy on held-out
    with torch.no_grad():
        img, ids = make_synth_batch(64, teacher_cfg, teacher_cfg.max_text_len, device)
        v = teacher.project(teacher.vision(img))
        logits = teacher.language(v, ids[:, :-1])
        acc = (logits.argmax(-1) == ids[:, 1:]).float().mean().item()
    print(f"      teacher token-accuracy ≈ {acc:.3f}")

    print("[2/3] training student with Switch-KD  (ℒ = ℒ_CE + λ_1·ℒ_Align + λ_2·ℒ_VSD) …")
    kd = SwitchKD(teacher, student, lambda_1=1.0, lambda_2=1.0, temperature=3.0, min_k=4)
    # only student params are trainable (teacher was frozen above)
    opt = torch.optim.AdamW(
        [p for p in kd.parameters() if p.requires_grad], lr=3e-3
    )

    history: list[float] = []
    vsd_history: list[float] = []
    for step in range(steps):
        image, text_ids = make_synth_batch(
            32, student_cfg, student_cfg.max_text_len, device
        )
        out = kd(image, text_ids[:, :-1], text_ids[:, 1:])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        history.append(out.loss.item())
        vsd_history.append(out.vsd.item())

        if step % 30 == 0 or step == steps - 1:
            print(
                f"  step {step:4d}  L={out.loss.item():.4f}  "
                f"CE={out.ce.item():.4f}  "
                f"Align={out.align.item():.4f}  "
                f"VSD={out.vsd.item():.4f}"
            )

    print("[3/3] evaluating agreement student↔teacher on held-out …")
    with torch.no_grad():
        img, ids = make_synth_batch(256, student_cfg, student_cfg.max_text_len, device)
        vT = teacher.project(teacher.vision(img))
        vS = student.project(student.vision(img))
        tlog = teacher.language(vT, ids[:, :-1])
        slog = student.language(vS, ids[:, :-1])
        agree = (tlog.argmax(-1) == slog.argmax(-1)).float().mean().item()
    print(f"      argmax agreement = {agree:.3f}")

    first10 = sum(history[:10]) / 10
    last10 = sum(history[-10:]) / 10
    vsd_first = sum(vsd_history[:10]) / 10
    vsd_last = sum(vsd_history[-10:]) / 10
    print(f"\nmean L   first 10 = {first10:.4f}  →  last 10 = {last10:.4f}")
    print(f"mean VSD first 10 = {vsd_first:.4f}  →  last 10 = {vsd_last:.4f}")
    assert last10 < first10, "Switch-KD did not reduce the loss — broken"
    assert vsd_last < vsd_first, "VSD term did not decrease — student vision is idle"
    assert agree > 0.3, f"agreement too low ({agree:.3f}) — distillation not working"
    print("✓ Switch-KD: total loss and VSD both decrease, student agrees with teacher.")


if __name__ == "__main__":
    main()
