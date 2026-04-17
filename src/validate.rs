use crate::model::Report;
use anyhow::{Result, anyhow};
use std::collections::BTreeSet;

pub struct ValidationReport {
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    pub fn is_fatal(&self) -> bool {
        !self.issues.is_empty()
    }
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "issues: {}, warnings: {}\n",
            self.issues.len(),
            self.warnings.len()
        ));
        for i in &self.issues {
            s.push_str(&format!("  [ERR] {i}\n"));
        }
        for w in &self.warnings {
            s.push_str(&format!("  [warn] {w}\n"));
        }
        s
    }
}

pub fn validate(report: &Report) -> ValidationReport {
    let mut issues: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    if report.papers.is_empty() {
        issues.push("no papers collected — pipeline produced an empty report".into());
    }

    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for p in &report.papers {
        if !seen.insert(&p.id) {
            issues.push(format!("duplicate paper id: {}", p.id));
        }
    }

    for p in &report.papers {
        if p.title.trim().is_empty() {
            issues.push(format!("paper {} has empty title", p.id));
        }
        if p.prompt.len() < 400 {
            issues.push(format!(
                "paper {} has suspiciously short prompt ({} chars)",
                p.id,
                p.prompt.len()
            ));
        }
        if !p.prompt.contains(&p.id) {
            issues.push(format!("paper {} prompt missing its own id", p.id));
        }
        if !p.prompt.contains(&p.sources.arxiv_abs) {
            issues.push(format!("paper {} prompt missing arxiv abs url", p.id));
        }
        if p.prompt.contains("(Abstract unavailable") {
            warnings.push(format!(
                "paper {} has no abstract — prompt will be less useful",
                p.id
            ));
        }
        if p.keywords.is_empty() {
            warnings.push(format!("paper {} produced zero keywords", p.id));
        }
    }

    let papers_with_abstracts = report
        .papers
        .iter()
        .filter(|p| !p.prompt.contains("(Abstract unavailable"))
        .count();
    if !report.papers.is_empty() {
        let ratio = papers_with_abstracts as f64 / report.papers.len() as f64;
        if ratio < 0.5 {
            issues.push(format!(
                "only {}/{} papers have real abstracts ({:.0}% < 50%) — scrape likely broken",
                papers_with_abstracts,
                report.papers.len(),
                ratio * 100.0
            ));
        }
    }

    ValidationReport { issues, warnings }
}

pub fn enforce(report: &Report) -> Result<ValidationReport> {
    let vr = validate(report);
    if vr.is_fatal() {
        return Err(anyhow!("validation failed:\n{}", vr.summary()));
    }
    Ok(vr)
}
