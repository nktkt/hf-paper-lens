use crate::model::{Paper, PaperSources, PromptPaper};
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::BTreeSet;

static WORD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z][A-Za-z0-9\-]{2,}").unwrap());

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "under", "which", "their",
    "these", "those", "such", "also", "have", "been", "were", "will", "when", "what", "where",
    "while", "about", "them", "they", "than", "then", "very", "much", "most", "more", "less",
    "some", "many", "each", "every", "other", "within", "upon", "using", "use", "can", "our",
    "via", "does", "has", "had", "its", "it's", "however", "therefore", "thus", "even", "both",
    "paper", "papers", "model", "models", "method", "methods", "results", "result", "study",
    "studies", "show", "shows", "showed", "propose", "proposed", "proposes", "present", "presents",
    "presented", "introduce", "introduces", "introduced", "approach", "approaches", "based",
    "achieve", "achieves", "achieved", "work", "works", "task", "tasks", "data", "set", "sets",
    "new", "novel", "first", "existing", "recent", "prior", "large", "small", "high", "low",
];

pub fn build_prompt(paper: &Paper) -> PromptPaper {
    let abstract_text = if paper.abstract_text.is_empty() {
        "(Abstract unavailable — inspect the source URLs for the full paper content.)"
            .to_string()
    } else {
        paper.abstract_text.clone()
    };

    let authors_line = if paper.authors.is_empty() {
        "(authors unlisted — see source)".to_string()
    } else {
        paper.authors.join(", ")
    };

    let keywords = extract_keywords(&paper.title, &paper.abstract_text, 8);
    let keywords_line = if keywords.is_empty() {
        String::new()
    } else {
        format!("\nKeywords: {}.\n", keywords.join(", "))
    };

    let prompt = format!(
        "You are an AI research assistant. Read the paper below and help the user understand, \
critique, and apply its ideas.\n\
\n\
[Paper]\n\
- Title: {title}\n\
- arXiv ID: {id}\n\
- Authors: {authors}\n\
- Hugging Face: {hf}\n\
- arXiv abstract: {abs}\n\
- arXiv PDF: {pdf}\n\
{kw}\n\
[Abstract]\n\
{abstract}\n\
\n\
[Your task]\n\
1. Summarize the paper in 3 bullet points (problem, method, result).\n\
2. Explain why this result is non-trivial in one paragraph.\n\
3. List 3 concrete follow-up questions a practitioner would ask.\n\
4. Propose one experiment, with dataset and metric, that would falsify the core claim.\n\
5. Extract any open-source artifacts (code, weights, datasets) the authors mention.\n\
\n\
Be concise and specific; cite the source URLs above when quoting.\n",
        title = paper.title,
        id = paper.id,
        authors = authors_line,
        hf = paper.hf_url,
        abs = paper.arxiv_abs_url,
        pdf = paper.arxiv_pdf_url,
        kw = keywords_line,
        abstract = abstract_text,
    );

    PromptPaper {
        id: paper.id.clone(),
        title: paper.title.clone(),
        prompt,
        keywords,
        sources: PaperSources {
            huggingface: paper.hf_url.clone(),
            arxiv_abs: paper.arxiv_abs_url.clone(),
            arxiv_pdf: paper.arxiv_pdf_url.clone(),
        },
    }
}

fn extract_keywords(title: &str, abstract_text: &str, limit: usize) -> Vec<String> {
    use std::collections::HashMap;
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut order: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let sources = [(title, 3u32), (abstract_text, 1u32)];
    for (src, weight) in sources {
        for m in WORD_RE.find_iter(src) {
            let w = m.as_str().to_lowercase();
            if w.len() < 4 || STOPWORDS.contains(&w.as_str()) {
                continue;
            }
            let entry = counts.entry(w.clone()).or_insert(0);
            *entry += weight;
            if seen.insert(w.clone()) {
                order.push(w);
            }
        }
    }
    let mut scored: Vec<(String, u32)> = order
        .into_iter()
        .map(|w| {
            let c = *counts.get(&w).unwrap_or(&0);
            (w, c)
        })
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.into_iter().take(limit).map(|(w, _)| w).collect()
}
