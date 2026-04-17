# Paper implementations

Reference implementations of distillation papers harvested by
`hf-paper-lens` on 2026-04-17.  Each one codes up the paper's *core
algorithmic novelty* in ~150 lines of plain PyTorch, with a runnable
demo (tiny models + synthetic data) that asserts the loss actually
decreases.  Both are designed to drop real Hugging Face models in by
swapping the `dummy` modules — the distillation logic itself is
model-agnostic.

| Paper | Folder | Run |
|---|---|---|
| [Cross-Tokenizer LLM Distillation through a Byte-Level Interface](https://arxiv.org/abs/2604.07466) | [`bld/`](bld/) | `python implementations/bld/train_bld.py` |
| [Switch-KD: Visual-Switch Knowledge Distillation for Vision-Language Models](https://arxiv.org/abs/2604.14629) | [`switch_kd/`](switch_kd/) | `python implementations/switch_kd/train_switch_kd.py` |

## Setup

```bash
# once, from repo root
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python torch
```

Then run the demos as shown in the table.  Both run on CPU in under a
minute and print a loss trajectory that verifiably decreases.

## Verified results (2026-04-17)

### BLD
- teacher and student use *different* tokenizers (shared UTF-8 base +
  distinct multi-byte BPE extensions)
- KL loss: **196.65 → 1.01** over 200 steps
- student is trained *only* via byte-level distillation (no hard labels)

### Switch-KD
- teacher (4-layer, 64-d) pretrained to 100 % token accuracy on
  synthetic buckets-to-caption task, then frozen
- student (2-layer, 64-d) trained purely via Switch-KD loss
- composite loss: **7.60 → 0.004** over 300 steps
- **argmax agreement teacher↔student = 1.000** on held-out batch

## Files you should read

Each paper's `README.md` lists the public API.  In one line:

- `implementations/bld/bld.py` — `bld_loss(teacher_token_logits, student_byte_logits, first_byte_matrix)`
- `implementations/switch_kd/switch_kd.py` — `SwitchKD(teacher, student)` wrapper and `dbild_loss(student_logits, teacher_logits)`.
