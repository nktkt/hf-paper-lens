# Switch-KD

Reference implementation of
**["Switch-KD: Visual-Switch Knowledge Distillation for Vision-Language Models"](https://arxiv.org/abs/2604.14629)**.

## What's here

- `switch_kd.py` — the paper's core contribution, independent of model choice:
  - `dbild_loss(student_logits, teacher_logits, top_k=32)` — Dynamic
    Bi-directional Logits Difference loss.  Forward KL restricted to
    teacher's top-k peaks + reverse KL restricted to student's top-k,
    weighted by a teacher-entropy schedule.
  - `SwitchKD(teacher, student)` — wraps a frozen teacher VLM and a
    trainable student VLM and implements the *visual-switch* trick:
    student's projected visual tokens are fed through the teacher's
    language decoder to produce a cross-modal probabilistic reference,
    which is distilled alongside the teacher's standard output.
- `train_switch_kd.py` — end-to-end runnable demo:
  - tiny teacher (96-d, 3 layers) and student (48-d, 2 layers) VLMs
  - synthetic "image bucket → caption palette" data
  - pretrain teacher, freeze, train student with Switch-KD only
  - asserts the loss decreases and argmax agreement > 0.3.

## Run

```bash
# from repo root
./.venv/bin/python implementations/switch_kd/train_switch_kd.py
```

Runs on CPU in ~30 s.

## Plugging in real VLMs

Any object conforming to the `VLM` protocol works:

```python
class VLM(Protocol):
    def vision(self, image): ...         # → (B, V_tok, D)
    def project(self, v): ...            # → (B, V_tok, D)
    def language(self, visual_tokens, text_ids): ...  # → (B, T, V_vocab)
```

This matches the TinyLLaVA / LLaVA decomposition the paper uses:
`vision_encoder → mm_projector → language_model`.  Wire your own three
methods to the teacher and student instances and pass them to
`SwitchKD(...)`.

## Requirements on teacher/student vocab

`dbild_loss` operates on aligned logit tensors, so **teacher and student
must share the output vocabulary** (standard VLM-KD assumption: both
start from the same language model backbone, only the vision/projector
side or depth differs).  If your pair has different vocabs, combine
this with the `bld.py` byte-level interface from
[`../bld/`](../bld/).
