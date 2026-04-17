# hf-paper-lens

A single-binary Rust CLI that harvests the **Hugging Face Daily Papers** feed,
deduplicates it, extracts each paper's metadata from Hugging Face and arXiv,
rewrites every paper as an **AI-ready prompt**, codifies the results into
machine-friendly artifacts (JSON, Markdown, and a compilable Rust module),
and validates the output before you ship it.

A separate `publish` subcommand creates a public GitHub repository from the
results in one shot.

## Why

Hugging Face's daily papers list is great for humans but annoying for
pipelines. You want, every day:

- One deduplicated list of today's papers (IDs, titles, URLs).
- A clean abstract per paper — no navigation chrome, no duplicated cards.
- A prompt per paper that an LLM can consume directly.
- A machine-readable snapshot you can diff, commit, or feed to a downstream
  tool.
- A check that none of the above is silently empty.

`hf-paper-lens` does all five, and nothing else.

## Features

- **Source of truth:** scrapes `https://huggingface.co/papers` (or
  `/papers/date/YYYY-MM-DD`) with a real User-Agent and exponential-backoff
  retries.
- **Deduplication:** arXiv IDs are canonical; the pipeline collapses
  duplicates both before and after detail fetching.
- **Per-paper analysis:** pulls title, abstract, authors, upvotes, submission
  date from the Hugging Face paper page; derives arXiv abstract and PDF URLs
  from the ID.
- **AI-ready prompts:** each paper becomes a templated prompt with a fixed
  5-step research-assistant task (summary, non-triviality, follow-up
  questions, falsification experiment, open-source artifacts).
- **Codified output:** JSON (for data pipelines), Markdown (for humans),
  individual `prompts/<id>.txt` files (for copy/paste), and a
  `papers.rs` Rust source module (compiles into downstream crates).
- **Validation:** the generated report is rejected if any paper has an empty
  title, a suspiciously short prompt, a missing URL, or if fewer than 50 % of
  papers have real abstracts (scrape regression guard).
- **One-shot publish:** `hf-paper-lens publish` calls `gh repo create
  --public --push`, giving you a public GitHub repo with the snapshot.

## Install

Requires Rust 1.75+ (2021 edition), plus `gh` and `git` for publishing.

```bash
git clone https://github.com/<you>/hf-paper-lens.git
cd hf-paper-lens
cargo build --release
./target/release/hf-paper-lens --help
```

## Usage

### Run the full pipeline

```bash
hf-paper-lens run --out out
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--date YYYY-MM-DD` | today (UTC) | Target Hugging Face daily-papers date. |
| `--out DIR` | `out` | Output directory. |
| `--concurrency N` | `6` | Concurrent detail fetches. |
| `--limit N` | `0` | Cap on papers (`0` = no cap). |

Output layout:

```
out/
├── papers.json           # structured snapshot (serde-friendly)
├── papers.md             # human-readable report with collapsible prompts
├── papers.rs             # compilable Rust module: `pub const PAPERS: &[Paper]`
└── prompts/
    ├── 2604.14228.txt    # one AI-ready prompt per paper
    └── ...
```

### Publish to GitHub

```bash
hf-paper-lens publish \
  --repo-dir . \
  --name hf-paper-lens \
  --description "Daily Hugging Face papers, prompted and codified."
```

This runs `git init`, commits everything, then
`gh repo create <name> --public --source . --remote origin --push`.

## Prompt format

Every paper is rewritten into a prompt of the form:

```text
You are an AI research assistant. Read the paper below and help the user
understand, critique, and apply its ideas.

[Paper]
- Title: ...
- arXiv ID: ...
- Authors: ...
- Hugging Face: https://huggingface.co/papers/<id>
- arXiv abstract: https://arxiv.org/abs/<id>
- arXiv PDF:     https://arxiv.org/pdf/<id>

Keywords: ...

[Abstract]
<cleaned abstract>

[Your task]
1. Summarize the paper in 3 bullet points (problem, method, result).
2. Explain why this result is non-trivial in one paragraph.
3. List 3 concrete follow-up questions a practitioner would ask.
4. Propose one experiment, with dataset and metric, that would falsify the
   core claim.
5. Extract any open-source artifacts (code, weights, datasets) the authors
   mention.

Be concise and specific; cite the source URLs above when quoting.
```

Deterministic and model-agnostic — works with any chat-completion LLM.

## Validation rules

The pipeline fails loudly if:

- The report is empty.
- Any paper ID appears twice.
- Any title is blank.
- Any prompt is under 400 characters, or is missing its own arXiv ID, or is
  missing the arXiv abstract URL.
- Fewer than 50 % of papers have real (non-placeholder) abstracts — a
  strong signal the upstream HTML changed.

Soft warnings are emitted for papers with no keywords or placeholder
abstracts; they do not block the run.

## Architecture

```
 index page         detail page (N×)         prompt builder
 ───────────   ──►  ────────────────    ──►  ──────────────   ──►  codify  ──►  validate
 /papers            /papers/<id>              model::PromptPaper        ↓
                                                                      out/{json,md,rs,prompts/}
```

- `src/fetch.rs` — HTTP + HTML extraction.
- `src/prompt.rs` — prompt + keyword construction.
- `src/codify.rs` — serializes the report to four artifact types.
- `src/validate.rs` — post-codify sanity checks.
- `src/publish.rs` — wraps `git` and `gh`.

## Limits

- Abstracts come from the Hugging Face paper page's `og:description` and
  visible paragraphs. PDFs are linked but not parsed here — they are exposed
  as `arxiv_pdf` so downstream tools (e.g. a vision-capable LLM) can ingest
  them directly.
- Upvote extraction is best-effort and depends on current HF markup.
- Hugging Face rate-limits aggressive scraping; the default concurrency (6)
  and exponential backoff are tuned to stay well under it.

## License

MIT — see [LICENSE](LICENSE).
