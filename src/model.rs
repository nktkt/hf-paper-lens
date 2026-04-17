use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Paper {
    pub id: String,
    pub title: String,
    pub hf_url: String,
    pub arxiv_abs_url: String,
    pub arxiv_pdf_url: String,
    pub authors: Vec<String>,
    pub abstract_text: String,
    pub upvotes: Option<u32>,
    pub submitted_date: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptPaper {
    pub id: String,
    pub title: String,
    pub prompt: String,
    pub keywords: Vec<String>,
    pub sources: PaperSources,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperSources {
    pub huggingface: String,
    pub arxiv_abs: String,
    pub arxiv_pdf: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub generated_at: String,
    pub target_date: String,
    pub total: usize,
    pub papers: Vec<PromptPaper>,
}
