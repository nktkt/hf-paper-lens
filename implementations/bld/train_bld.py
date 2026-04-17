"""
End-to-end demo training a student LM with BLD (faithful to eqs. 3 & 5
of https://arxiv.org/abs/2604.07466).

Pipeline per batch:
  1. Tokenize synthetic text with *different* teacher and student
     tokenizers (`ByteTokenizer`).
  2. Pre-trained frozen teacher produces per-token probabilities
     conditioned on teacher history.
  3. For each student token t_ℓ with bytes b^ℓ_1..b^ℓ_{n_ℓ} (truncated
     to 10), compute teacher's conditional byte distribution at each
     intra-token position j via the covering-based marginalizer
     (TeacherByteMarginalizer) using the teacher's hidden state at the
     LAST committed teacher-token boundary ≤ (start of b^ℓ_j).
  4. Student LM emits token logits and 10 parallel byte logits per
     position.
  5. Loss = CE_token + 1.0 · CE_byte + 0.1 · KL_byte   (Table 4).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bld import (
    BYTE_PAD,
    BYTE_VOCAB_SIZE,
    DEFAULT_LAMBDA_B,
    DEFAULT_LAMBDA_KL,
    MAX_BYTES_PER_TOKEN,
    ByteDecoderHead,
    TeacherByteMarginalizer,
    bld_total_loss,
)


# ---------------------------------------------------------------------------
# Two minimal tokenizers over UTF-8 bytes, with *different* extra tokens
# ---------------------------------------------------------------------------


class ByteTokenizer:
    """Vocab: index 0 = PAD, 1..256 = raw bytes, plus user-supplied
    multi-byte extras.  Greedy longest-match."""

    PAD_ID = 0

    def __init__(self, extra_tokens: list[bytes]):
        base: list[bytes] = [b""]
        base.extend(bytes([b]) for b in range(256))
        self.vocab: list[bytes] = base + extra_tokens
        self._match_order = sorted(
            range(len(self.vocab)), key=lambda i: -len(self.vocab[i])
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        out: list[int] = []
        i = 0
        while i < len(data):
            for idx in self._match_order:
                tok = self.vocab[idx]
                if len(tok) == 0:
                    continue
                if data[i : i + len(tok)] == tok:
                    out.append(idx)
                    i += len(tok)
                    break
            else:
                out.append(data[i] + 1)
                i += 1
        return out


# ---------------------------------------------------------------------------
# Tiny causal Transformer LM
# ---------------------------------------------------------------------------


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 128


class DummyLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok(ids) + self.pos(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        h = self.blocks(x, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.lm_head(h), h


# ---------------------------------------------------------------------------
# Per-batch BLD target construction (eq. 3 applied to every student byte)
# ---------------------------------------------------------------------------


def build_bld_targets_for_example(
    text_bytes: bytes,
    student_ids: list[int],
    student_tk: ByteTokenizer,
    teacher_ids: list[int],
    teacher_tk: ByteTokenizer,
    teacher_token_probs: torch.Tensor,   # (T_teacher, V_T)
    marginalizer: TeacherByteMarginalizer,
    max_bytes: int = MAX_BYTES_PER_TOKEN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce per-student-token byte targets + teacher byte distributions.

    Returns:
        target_bytes        (T_s, max_bytes) — byte ids in [0, 255] or BYTE_PAD
        teacher_byte_probs  (T_s, max_bytes, BYTE_VOCAB_SIZE)
        target_next_token   (T_s,)          — ids of t_{ℓ+1} with -100 padding
    """
    # map: absolute byte index in full text → (teacher_token_idx, intra_byte_pos)
    # we build a byte→teacher_token map by walking teacher_ids.
    teacher_byte_starts: list[int] = []  # absolute byte index where each teacher token starts
    cursor = 0
    for tid in teacher_ids:
        teacher_byte_starts.append(cursor)
        cursor += len(teacher_tk.vocab[tid])
    total_teacher_bytes = cursor

    T_s = len(student_ids)
    target_bytes = torch.full(
        (T_s, max_bytes), BYTE_PAD, dtype=torch.long
    )
    teacher_byte_probs = torch.zeros(T_s, max_bytes, BYTE_VOCAB_SIZE)
    target_next_token = torch.full((T_s,), -100, dtype=torch.long)

    # Absolute byte cursor as we walk student tokens
    student_cursor = 0
    for ell, sid in enumerate(student_ids):
        sbs = student_tk.vocab[sid]
        n_l = min(len(sbs), max_bytes)

        # Record target bytes for this student token
        for j in range(n_l):
            target_bytes[ell, j] = sbs[j]

        # Record next-token target if available
        if ell + 1 < T_s:
            target_next_token[ell] = student_ids[ell + 1]

        # For each byte position j ∈ {0..n_l-1} of this student token,
        # compute P_T(b^ℓ_j | b^ℓ_<j, t_<ℓ) via covering eval.
        for j in range(n_l):
            abs_byte_idx = student_cursor + j
            if abs_byte_idx >= total_teacher_bytes:
                teacher_byte_probs[ell, j, :256] = 1.0 / 256.0
                continue
            # find last committed teacher token: the last teacher token
            # that *ends* strictly before abs_byte_idx.
            committed_idx = -1
            for k, start in enumerate(teacher_byte_starts):
                end = start + len(teacher_tk.vocab[teacher_ids[k]])
                if end <= abs_byte_idx:
                    committed_idx = k
                else:
                    break
            # teacher history length
            history_len = committed_idx + 1
            if history_len == 0:
                # no committed tokens ⇒ use uniform prior
                teacher_byte_probs[ell, j, :256] = 1.0 / 256.0
                continue
            # intra-token prefix: bytes produced so far in the
            # in-progress teacher token
            committed_end = (
                teacher_byte_starts[committed_idx]
                + len(teacher_tk.vocab[teacher_ids[committed_idx]])
            )
            intra_prefix = bytes(
                text_bytes[committed_end : abs_byte_idx]
            )
            # teacher_token_probs[history_len - 1] is the distribution
            # over the NEXT teacher token given t_<(history_len).
            # We want distribution after history of length history_len,
            # i.e. teacher_token_probs[history_len - 1].
            probs_row = teacher_token_probs[history_len - 1]  # (V_T,)
            teacher_byte_probs[ell, j] = marginalizer.marginal_given_prefix(
                probs_row, intra_prefix
            )

        student_cursor += len(sbs)

    return target_bytes, teacher_byte_probs, target_next_token


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------


CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "knowledge distillation transfers capability from teacher to student",
    "byte level interfaces are tokenizer agnostic common ground",
    "vision language models benefit from multimodal alignment",
    "transformers remain the dominant neural architecture",
    "softmax outputs span a probability simplex",
    "cross tokenizer distillation was long considered painful",
    "every sequence is ultimately a sequence of bytes",
]


def sample_text(max_bytes: int) -> str:
    return random.choice(CORPUS)[:max_bytes]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def pretrain_teacher(
    teacher: DummyLM,
    teacher_tk: ByteTokenizer,
    device: str,
    steps: int = 300,
    batch_size: int = 16,
    seq_len: int = 48,
) -> None:
    opt = torch.optim.AdamW(teacher.parameters(), lr=3e-3)
    teacher.train()
    for _ in range(steps):
        texts = [sample_text(seq_len) for _ in range(batch_size)]
        ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            seq = teacher_tk.encode(t)[:seq_len]
            ids[i, : len(seq)] = torch.tensor(seq, device=device)
        logits, _ = teacher(ids[:, :-1])
        target = ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, teacher_tk.vocab_size),
            target.reshape(-1),
            ignore_index=ByteTokenizer.PAD_ID,
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)


def main(
    steps: int = 200,
    batch_size: int = 6,
    seq_len_bytes: int = 48,
    seed: int = 0,
    device: str = "cpu",
) -> None:
    torch.manual_seed(seed)
    random.seed(seed)

    teacher_extras = [b"the ", b"ion", b"ing", b"tion", b" the", b"er ", b" a "]
    student_extras = [b"is ", b"ns ", b"ed ", b"ly ", b" to ", b"ment"]
    tk_T = ByteTokenizer(teacher_extras)
    tk_S = ByteTokenizer(student_extras)

    teacher = DummyLM(LMConfig(vocab_size=tk_T.vocab_size)).to(device)
    student = DummyLM(LMConfig(vocab_size=tk_S.vocab_size)).to(device)
    byte_head = ByteDecoderHead(hidden_dim=student.cfg.d_model).to(device)

    print("[1/3] pretraining teacher …")
    pretrain_teacher(teacher, tk_T, device)

    # Marginalizer over teacher vocab
    marginalizer = TeacherByteMarginalizer(tk_T.vocab)

    opt = torch.optim.AdamW(
        list(student.parameters()) + list(byte_head.parameters()), lr=3e-3
    )

    print("[2/3] training student with full BLD loss (CE + λ_b·CE_byte + λ_KL·KL_byte) …")
    history_total: list[float] = []
    history_kl: list[float] = []

    for step in range(steps):
        texts = [sample_text(seq_len_bytes) for _ in range(batch_size)]
        # Tokenize per side
        ids_T_list = [tk_T.encode(t) for t in texts]
        ids_S_list = [tk_S.encode(t) for t in texts]

        T_s = max(len(x) for x in ids_S_list)
        T_t = max(len(x) for x in ids_T_list)
        ids_S = torch.zeros(batch_size, T_s, dtype=torch.long, device=device)
        ids_T = torch.zeros(batch_size, T_t, dtype=torch.long, device=device)
        for i, (s, t) in enumerate(zip(ids_S_list, ids_T_list)):
            ids_S[i, : len(s)] = torch.tensor(s, device=device)
            ids_T[i, : len(t)] = torch.tensor(t, device=device)

        # Teacher forward (frozen, no grad)
        with torch.no_grad():
            t_logits, _ = teacher(ids_T)
            t_probs = t_logits.softmax(dim=-1)  # (B, T_t, V_T)

        # Per-example BLD target construction (eq. 3)
        target_bytes_batch = torch.full(
            (batch_size, T_s, MAX_BYTES_PER_TOKEN), BYTE_PAD, dtype=torch.long
        )
        teacher_byte_probs_batch = torch.zeros(
            batch_size, T_s, MAX_BYTES_PER_TOKEN, BYTE_VOCAB_SIZE
        )
        target_next_token_batch = torch.full(
            (batch_size, T_s), -100, dtype=torch.long
        )
        for i, text in enumerate(texts):
            t_bytes, t_probs_row, t_next = build_bld_targets_for_example(
                text_bytes=text.encode("utf-8"),
                student_ids=ids_S_list[i],
                student_tk=tk_S,
                teacher_ids=ids_T_list[i],
                teacher_tk=tk_T,
                teacher_token_probs=t_probs[i, : len(ids_T_list[i])].cpu(),
                marginalizer=marginalizer,
            )
            n = t_bytes.size(0)
            target_bytes_batch[i, :n] = t_bytes
            teacher_byte_probs_batch[i, :n] = t_probs_row
            target_next_token_batch[i, :n] = t_next
        target_bytes_batch = target_bytes_batch.to(device)
        teacher_byte_probs_batch = teacher_byte_probs_batch.to(device)
        target_next_token_batch = target_next_token_batch.to(device)

        # Student forward
        s_logits, s_hidden = student(ids_S)
        s_byte_logits = byte_head(s_hidden)  # (B, T_s, 10, 260)

        # Loss
        losses = bld_total_loss(
            student_token_logits=s_logits,
            student_byte_logits=s_byte_logits,
            student_target_tokens=target_next_token_batch,
            student_target_bytes=target_bytes_batch,
            teacher_byte_targets=teacher_byte_probs_batch,
            lambda_b=DEFAULT_LAMBDA_B,
            lambda_kl=DEFAULT_LAMBDA_KL,
        )

        opt.zero_grad()
        losses.total.backward()
        opt.step()

        history_total.append(losses.total.item())
        history_kl.append(losses.kl_byte.item())

        if step % 20 == 0 or step == steps - 1:
            print(
                f"  step {step:4d}  L={losses.total.item():.4f}  "
                f"CE_tok={losses.ce_token.item():.4f}  "
                f"CE_byte={losses.ce_byte.item():.4f}  "
                f"KL_byte={losses.kl_byte.item():.4f}"
            )

    first10 = sum(history_total[:10]) / 10
    last10 = sum(history_total[-10:]) / 10
    kl_first = sum(history_kl[:10]) / 10
    kl_last = sum(history_kl[-10:]) / 10
    print(f"\nmean L  first 10 = {first10:.4f}  →  last 10 = {last10:.4f}")
    print(f"mean KL first 10 = {kl_first:.4f}  →  last 10 = {kl_last:.4f}")
    assert last10 < first10, "total BLD loss did not decrease"
    assert kl_last < kl_first, "byte-KL did not decrease — covering term is idle"
    print("✓ BLD: both total loss and the KL_byte term decrease.")


if __name__ == "__main__":
    main()
