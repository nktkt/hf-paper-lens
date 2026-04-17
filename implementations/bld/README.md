# Byte-Level Distillation (BLD)

Reference implementation of
**["Cross-Tokenizer LLM Distillation through a Byte-Level Interface"](https://arxiv.org/abs/2604.07466)**.

## What's here

- `bld.py` — the paper's core contribution, independent of model choice:
  - `build_first_byte_matrix(vocab)` — vocab-agnostic projection
    M ∈ R^{|V|×256}, M[v, b] = 1 iff the first UTF-8 byte of token `v` is `b`.
  - `token_logits_to_byte_probs(logits, M)` — marginalizes teacher
    subword logits onto a 256-way next-byte distribution.
  - `ByteDecoderHead` — a 2-layer MLP added to the student; reads the
    student's hidden state and outputs 256 byte logits.
  - `bld_loss(teacher_token_logits, student_byte_logits, M)` — forward
    KL between teacher's byte-marginal and student's byte distribution.
- `train_bld.py` — a self-contained demo:
  - two minimal tokenizers with *different* extra vocab entries
  - a tiny transformer LM (64-d, 2 layers) for each side
  - pre-trains the teacher on next-token prediction, then freezes it
  - trains the student + byte head with BLD **only** (no label
    supervision) and asserts the KL decreases.

## Run

```bash
# from repo root
./.venv/bin/python implementations/bld/train_bld.py
```

Runs on CPU in ~15 s.  You should see the loss fall from ~5.5 to ~3.0.

## Plugging in real HF models

`DummyLM` just needs to produce `logits` of shape `(B, T, V)` and a
hidden state `(B, T, d)`.  Any `AutoModelForCausalLM` does that — swap
it in, provide each tokenizer's vocab as a `list[bytes]`, and everything
else carries over.
