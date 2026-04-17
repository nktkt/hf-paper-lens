"""
Switch-KD — faithful to the equations in
https://arxiv.org/abs/2604.14629 (§3, §4, Eqs. 1, 4, 5, 6, 7, 8–9, 13, 14, 15).

Component notation (paper §3):
    Teacher VLM  M^T = (V^T, P^T, L^T)     (vision, projector, language)
    Student VLM  M^S = (V^S, P^S, L^S)
    x_v : image,  x_t : text ids

Pathway logits:
    Standard teacher:        z^T      = L^T( P^T( V^T(x_v) ), x_t )
    Standard student:        z^S      = L^S( P^S( V^S(x_v) ), x_t )
    Visual-switch (Eq. 5):   z^Switch = L^T( P^T( V^S(x_v) ), x_t )
        — student's vision goes through the *teacher's* frozen
          projector and language model.

Losses:
    Eq. 4 (standard alignment):   ℒ_Align = ℒ_DBiLD(z^T, z^S)
    Eq. 6 (visual-switch):        ℒ_VSD   = ℒ_DBiLD(z^T, z^Switch)
    Eq. 7 (DBiLD decomposition):  ℒ_DBiLD = ℒ_t + ℒ_s
    Eq. 14 (teacher-led):         ℒ_t = D_RKL[ p^t_led  ‖ p^s_cor ]
    Eq. 15 (student-led):         ℒ_s = D_RKL[ p^t_cor ‖ p^s_led ]
    Eq. 1 (total):                ℒ   = ℒ_CE + λ_1 ℒ_Align + λ_2 ℒ_VSD
        with λ_1 = λ_2 = 1.0 and temperature τ = 3   (Table 4).

Dynamic top-k (Eqs. 8–9): paper uses the Kneedle algorithm to detect the
"knee" that separates peaks from the long tail.  We implement Kneedle
directly on the sorted-descending probability curve per logit row:
                 knee = argmax_i  ( (1 - i/(V-1)) - ŷ_i )
where ŷ is the min-max-normalized sorted probability curve.

D_RKL (Eq. 13 convention): reverse KL using the *second* argument as the
target distribution, i.e. D_RKL[A ‖ B] := KL(B ‖ A).  Optimizing ℒ_t
therefore pushes p^s_cor toward p^t_led, the teacher-led leading
distribution, while ℒ_s pushes p^s_led toward p^t_cor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TEMPERATURE = 3.0      # τ in Table 4
DEFAULT_LAMBDA_1 = 1.0          # λ_1 for ℒ_Align
DEFAULT_LAMBDA_2 = 1.0          # λ_2 for ℒ_VSD


class VLM(Protocol):
    def vision(self, image: torch.Tensor) -> torch.Tensor: ...
    def project(self, v: torch.Tensor) -> torch.Tensor: ...
    def language(
        self, visual_tokens: torch.Tensor, text_ids: torch.Tensor
    ) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Dynamic top-k via Kneedle (Eqs. 8–9)
# ---------------------------------------------------------------------------


def kneedle_topk_mask(
    logits: torch.Tensor,
    min_k: int = 4,
    max_k_frac: float = 0.5,
) -> torch.Tensor:
    """Per-row Kneedle knee detection on the sorted-descending prob curve.

    Returns a boolean mask of the same shape as `logits`, with 1s at the
    top-k indices where k is chosen per row as (knee + 1).
    """
    probs = logits.softmax(dim=-1)
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    V = sorted_probs.size(-1)
    max_k = max(min_k + 1, int(V * max_k_frac))

    x_norm = torch.linspace(
        0, 1, V, device=logits.device, dtype=logits.dtype
    )
    y_min = sorted_probs.amin(dim=-1, keepdim=True)
    y_max = sorted_probs.amax(dim=-1, keepdim=True)
    y_norm = (sorted_probs - y_min) / (y_max - y_min).clamp_min(1e-9)

    # For a descending curve, the knee is farthest below the chord
    # connecting endpoints (0,1) and (1,0).  Distance to that line:
    distance = (1 - x_norm) - y_norm  # broadcast over leading dims
    # exclude degenerate very-early and very-late positions
    distance[..., :min_k] = -float("inf")
    distance[..., max_k:] = -float("inf")
    knee = distance.argmax(dim=-1)  # (...,)

    # k = knee + 1 positions included
    k_values = (knee + 1).unsqueeze(-1)
    # rank[..., v] = position of v in sorted order
    rank = sorted_idx.argsort(dim=-1)
    mask = (rank < k_values).to(logits.dtype)
    return mask


# ---------------------------------------------------------------------------
# DBiLD loss  (Eqs. 7, 14, 15)
# ---------------------------------------------------------------------------


def _masked_softmax(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = z.masked_fill(mask == 0, float("-inf"))
    return masked.softmax(dim=-1)


def _reverse_kl(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """D_RKL[A ‖ B] := KL(B ‖ A) = Σ B ( log B − log A )."""
    return (B * (B.clamp_min(eps).log() - A.clamp_min(eps).log())).sum(dim=-1)


def dbild_loss(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    min_k: int = 4,
) -> torch.Tensor:
    """ℒ_DBiLD = ℒ_t + ℒ_s."""
    zt = logits_teacher / temperature
    zs = logits_student / temperature

    # Teacher-led (Eq. 14) — mask from teacher's own distribution
    mask_T = kneedle_topk_mask(zt.detach(), min_k=min_k)
    # Student-led (Eq. 15) — mask from student's distribution (detached for
    # mask computation so the mask itself is not part of the graph).
    mask_S = kneedle_topk_mask(zs.detach(), min_k=min_k)

    p_t_led = _masked_softmax(zt, mask_T)
    p_s_cor = _masked_softmax(zs, mask_T)
    p_t_cor = _masked_softmax(zt, mask_S)
    p_s_led = _masked_softmax(zs, mask_S)

    L_t = _reverse_kl(p_t_led, p_s_cor).mean()   # Eq. 14
    L_s = _reverse_kl(p_t_cor, p_s_led).mean()   # Eq. 15
    return L_t + L_s


# ---------------------------------------------------------------------------
# Switch-KD: Eqs. 1, 4, 5, 6 put together
# ---------------------------------------------------------------------------


@dataclass
class SwitchKDOutput:
    loss: torch.Tensor
    ce: torch.Tensor
    align: torch.Tensor
    vsd: torch.Tensor


class SwitchKD(nn.Module):
    """Wraps a frozen teacher VLM and a trainable student VLM.

    forward(image, text_input_ids, text_target_ids) returns:
        ℒ = ℒ_CE + λ_1 ℒ_Align + λ_2 ℒ_VSD   (Eq. 1)

    Assumes teacher and student share the output vocabulary (standard
    VLM-KD setup), and that the teacher's projector accepts the
    student's vision features as input — in the paper this holds because
    both use the same vision backbone (e.g. CLIP-ViT-L/14) and the
    language backbones share hidden size.  For models where this is not
    true, add an adapter before `teacher.project(v_student)`.
    """

    def __init__(
        self,
        teacher: VLM,
        student: VLM,
        lambda_1: float = DEFAULT_LAMBDA_1,
        lambda_2: float = DEFAULT_LAMBDA_2,
        temperature: float = DEFAULT_TEMPERATURE,
        min_k: int = 4,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.temperature = temperature
        self.min_k = min_k
        # Freeze every teacher parameter.
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        image: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_target_ids: torch.Tensor,
    ) -> SwitchKDOutput:
        # ---- z^S: student's own path (fully trainable)
        v_student = self.student.vision(image)
        v_proj_S = self.student.project(v_student)
        z_S = self.student.language(v_proj_S, text_input_ids)

        # ---- z^Switch: student's vision, teacher's frozen projector+LM
        # Gradients flow only to student.vision (teacher params are frozen).
        v_proj_switch = self.teacher.project(v_student)
        z_Switch = self.teacher.language(v_proj_switch, text_input_ids)

        # ---- z^T: teacher's own path (fully detached)
        with torch.no_grad():
            v_teacher = self.teacher.vision(image)
            v_proj_T = self.teacher.project(v_teacher)
            z_T = self.teacher.language(v_proj_T, text_input_ids)

        V = z_S.size(-1)
        L_CE = F.cross_entropy(
            z_S.reshape(-1, V),
            text_target_ids.reshape(-1),
            ignore_index=-100,
        )
        L_Align = dbild_loss(z_T, z_S, temperature=self.temperature, min_k=self.min_k)
        L_VSD = dbild_loss(
            z_T, z_Switch, temperature=self.temperature, min_k=self.min_k
        )

        total = L_CE + self.lambda_1 * L_Align + self.lambda_2 * L_VSD
        return SwitchKDOutput(
            loss=total,
            ce=L_CE.detach(),
            align=L_Align.detach(),
            vsd=L_VSD.detach(),
        )
