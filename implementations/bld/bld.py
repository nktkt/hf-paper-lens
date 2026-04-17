"""
Byte-Level Distillation (BLD) — faithful to the equations in
https://arxiv.org/abs/2604.07466 (§3.1 / §3.2, Tables 3–4).

Notation (paper):
    V_T              : teacher vocabulary, tokens v have bytes(v) ∈ bytes*
    𝒯_S              : student tokenizer
    t_ℓ              : the ℓ-th student token, with bytes b^ℓ_1 … b^ℓ_{n_ℓ}
    f_S(t_<ℓ)        : student LM's next-token distribution over V_S
    f^b_S(t_<ℓ, j)   : student byte head's distribution at intra-token
                       position j ∈ {1 … 10}, reading h^S_{ℓ-1}
    P_T(·)           : teacher distribution

Equation 3 (conditional byte probability):
        P_T(b_i | b_<i) = P_T({b_1..b_i}) / P_T({b_1..b_{i-1}})
    where P_T({b_1..b_i}) is the total probability mass of all teacher
    token sequences whose concatenated bytes equal b_1..b_i.

    Exact evaluation is intractable in general; we use the canonical
    "greedy-coverage" approximation that the paper itself references:
    committed teacher tokens up to the byte just before the in-progress
    one act as the history, and the covering is restricted to tokens
    whose byte prefix matches the current intra-token prefix.  This
    yields, for a committed history h(s) and intra-token prefix prefix:

        P_T(next byte = B | h(s), prefix) =
              Σ_{v : bytes(v) starts with prefix++B}  P_T(v | h(s))
            / Σ_{v : bytes(v) starts with prefix}     P_T(v | h(s))

    which at prefix = b"" recovers the first-byte marginal.

Equation 5 (total loss):
        L = CE(δ(t_ℓ), f_S(t_<ℓ))
          + λ_b  · Σ_{j ≤ min(n_ℓ, 10)} CE(δ(b^ℓ_j), f^b_S(t_<ℓ, j))
          + λ_KL · Σ_{j ≤ min(n_ℓ, 10)} KL( P_T(b^ℓ_j | b^ℓ_<j, t_<ℓ)
                                            || f^b_S(t_<ℓ, j) )
    with λ_b = 1.0, λ_KL = 0.1 (Table 4).

Architecture:
    f^b_S = a bank of 10 parallel linear projections from the student's
    hidden dimension to a 260-way byte vocabulary (Section 3.2, "a
    simple linear projection ... 10 parallel linear projections").
    Tokens with n_ℓ > 10 are supervised only on their first 10 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

BYTE_VOCAB_SIZE = 260            # 256 raw bytes + 4 special slots (paper §3.2)
MAX_BYTES_PER_TOKEN = 10         # paper §3.2: 10 parallel byte heads
DEFAULT_LAMBDA_B = 1.0           # Table 4
DEFAULT_LAMBDA_KL = 0.1          # Table 4
BYTE_PAD = 256                   # one of the 4 special byte slots: pad


# ---------------------------------------------------------------------------
# Student byte head: 10 parallel linear projections (paper §3.2)
# ---------------------------------------------------------------------------


class ByteDecoderHead(nn.Module):
    """10 parallel linear projections: h → byte-vocab(260) at each of
    the first 10 byte positions inside the predicted token.  Non
    autoregressive over bytes; every byte position is produced from the
    same hidden state in parallel."""

    def __init__(
        self,
        hidden_dim: int,
        byte_vocab_size: int = BYTE_VOCAB_SIZE,
        max_bytes: int = MAX_BYTES_PER_TOKEN,
    ):
        super().__init__()
        self.max_bytes = max_bytes
        self.byte_vocab_size = byte_vocab_size
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, byte_vocab_size) for _ in range(max_bytes)]
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, D).  Returns (B, T, max_bytes, byte_vocab_size)."""
        return torch.stack([head(h) for head in self.heads], dim=-2)


# ---------------------------------------------------------------------------
# Teacher byte-marginalization: eq. 3 (greedy-coverage evaluation)
# ---------------------------------------------------------------------------


class TeacherByteMarginalizer:
    """Computes P_T(b_i | b_<i) for every byte position in a sequence,
    given per-token teacher probabilities and the canonical teacher
    tokenization of the full byte sequence.

    Usage:
        m = TeacherByteMarginalizer(teacher_vocab)
        teacher_token_probs = softmax(teacher_logits)      # (B, T_tok, V_T)
        byte_probs = m.byte_distributions(
            teacher_token_probs,                             # (B, T_tok, V_T)
            teacher_token_ids,                               # (B, T_tok)
            target_byte_positions,                           # list per sample
        )
        # → (B, Nbytes, byte_vocab_size) aligned with target_byte_positions
    """

    def __init__(self, teacher_vocab: Sequence[bytes]):
        self.vocab: list[bytes] = list(teacher_vocab)
        self.V = len(self.vocab)

        # For the covering evaluation, we need, for each possible
        # intra-token prefix, the list of (token_id, next_byte).  We
        # build this incrementally keyed on prefix length.
        # prefix_index[k] maps a k-byte prefix → tensor of
        #     (token_id, next_byte) pairs where next_byte is bytes(v)[k].
        self._by_prefix: dict[bytes, list[tuple[int, int]]] = {}
        for v, bs in enumerate(self.vocab):
            for k in range(len(bs)):
                self._by_prefix.setdefault(bs[:k], []).append((v, bs[k]))

    def marginal_given_prefix(
        self,
        token_probs_at_history: torch.Tensor,  # (V_T,)
        intra_token_prefix: bytes,
    ) -> torch.Tensor:
        """Return P_T(next byte | committed history, intra-token prefix):
        shape (byte_vocab_size,)."""
        candidates = self._by_prefix.get(intra_token_prefix)
        out = torch.zeros(BYTE_VOCAB_SIZE, device=token_probs_at_history.device)
        if not candidates:
            # intra-token prefix not realizable by any teacher token ⇒
            # flat distribution (rare, but well-defined fallback).
            out[:256] = 1.0 / 256.0
            return out
        total = 0.0
        for tok_id, next_byte in candidates:
            p = token_probs_at_history[tok_id].item()
            out[next_byte] += p
            total += p
        if total > 0:
            out = out / total
        else:
            out[:256] = 1.0 / 256.0
        return out


# ---------------------------------------------------------------------------
# Loss assembly: equation 5
# ---------------------------------------------------------------------------


@dataclass
class BLDLosses:
    total: torch.Tensor
    ce_token: torch.Tensor
    ce_byte: torch.Tensor
    kl_byte: torch.Tensor


def bld_total_loss(
    student_token_logits: torch.Tensor,      # (B, T_s, V_S)
    student_byte_logits: torch.Tensor,       # (B, T_s, 10, byte_vocab_size)
    student_target_tokens: torch.Tensor,     # (B, T_s)
    student_target_bytes: torch.Tensor,      # (B, T_s, 10)     BYTE_PAD where absent
    teacher_byte_targets: torch.Tensor,      # (B, T_s, 10, byte_vocab_size) probs
    lambda_b: float = DEFAULT_LAMBDA_B,
    lambda_kl: float = DEFAULT_LAMBDA_KL,
) -> BLDLosses:
    """Equation 5, mean-reduced across batch and valid positions."""
    B, Ts, V_S = student_token_logits.shape
    _, _, J, Bv = student_byte_logits.shape

    # (a) token CE — ignore padding (-100 convention)
    ce_token = F.cross_entropy(
        student_token_logits.reshape(-1, V_S),
        student_target_tokens.reshape(-1),
        ignore_index=-100,
    )

    # (b) byte CE — per-position, ignore where byte is BYTE_PAD
    ce_byte = F.cross_entropy(
        student_byte_logits.reshape(-1, Bv),
        student_target_bytes.reshape(-1),
        ignore_index=BYTE_PAD,
    )

    # (c) byte KL — KL(teacher_byte || student_byte), ignore padded slots
    log_q = F.log_softmax(student_byte_logits, dim=-1)
    # build a mask: valid if student_target_bytes != BYTE_PAD
    mask = (student_target_bytes != BYTE_PAD).float().unsqueeze(-1)  # (...,1)
    kl_per_slot = (
        teacher_byte_targets
        * (teacher_byte_targets.clamp_min(1e-12).log() - log_q)
    ).sum(dim=-1)  # (B, Ts, 10)
    kl_byte = (kl_per_slot * mask.squeeze(-1)).sum() / mask.sum().clamp_min(1.0)

    total = ce_token + lambda_b * ce_byte + lambda_kl * kl_byte
    return BLDLosses(
        total=total,
        ce_token=ce_token.detach(),
        ce_byte=ce_byte.detach(),
        kl_byte=kl_byte.detach(),
    )
