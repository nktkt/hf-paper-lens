use anyhow::{Context, Result, anyhow};
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use std::collections::BTreeSet;
use std::time::Duration;

use crate::model::Paper;

static PAPER_PATH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^/papers/(\d{4}\.\d{4,6})(?:/|$)").unwrap());

pub fn build_client() -> Result<Client> {
    Ok(Client::builder()
        .user_agent("hf-paper-lens/0.1 (+https://github.com/nktkt/hf-paper-lens)")
        .timeout(Duration::from_secs(30))
        .connect_timeout(Duration::from_secs(10))
        .build()?)
}

fn first_text(doc: &Html, selector: &str) -> Option<String> {
    let sel = Selector::parse(selector).ok()?;
    doc.select(&sel)
        .next()
        .map(|el| el.text().collect::<String>().trim().to_string())
        .filter(|s| !s.is_empty())
}

fn meta_content(doc: &Html, property_or_name: &str) -> Option<String> {
    let sel = Selector::parse(&format!(
        "meta[property=\"{prop}\"], meta[name=\"{prop}\"]",
        prop = property_or_name
    ))
    .ok()?;
    doc.select(&sel)
        .next()
        .and_then(|el| el.value().attr("content"))
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub async fn fetch_paper_ids(client: &Client, date: Option<&str>) -> Result<Vec<String>> {
    let url = match date {
        Some(d) => format!("https://huggingface.co/papers/date/{d}"),
        None => "https://huggingface.co/papers".to_string(),
    };
    let html = fetch_with_retry(client, &url, 4).await?;
    let doc = Html::parse_document(&html);
    let a_sel = Selector::parse("a[href^=\"/papers/\"]").unwrap();
    let mut ids: BTreeSet<String> = BTreeSet::new();
    for a in doc.select(&a_sel) {
        if let Some(href) = a.value().attr("href") {
            if let Some(caps) = PAPER_PATH_RE.captures(href) {
                ids.insert(caps[1].to_string());
            }
        }
    }
    if ids.is_empty() {
        return Err(anyhow!(
            "no paper ids extracted from {url} — the page layout may have changed"
        ));
    }
    Ok(ids.into_iter().collect())
}

pub async fn fetch_paper_detail(client: &Client, id: &str) -> Result<Paper> {
    let hf_url = format!("https://huggingface.co/papers/{id}");
    let html = fetch_with_retry(client, &hf_url, 4).await?;
    let doc = Html::parse_document(&html);

    let title = meta_content(&doc, "og:title")
        .or_else(|| first_text(&doc, "h1"))
        .map(|t| strip_title_prefix(&t))
        .unwrap_or_else(|| format!("arXiv:{id}"));

    let abstract_text = extract_abstract(&doc);

    let authors = extract_authors(&doc);

    let upvotes = extract_upvotes(&doc);

    let submitted_date = extract_submitted_date(&doc);

    Ok(Paper {
        id: id.to_string(),
        title: clean_text(&title),
        hf_url: hf_url.clone(),
        arxiv_abs_url: format!("https://arxiv.org/abs/{id}"),
        arxiv_pdf_url: format!("https://arxiv.org/pdf/{id}"),
        authors,
        abstract_text,
        upvotes,
        submitted_date,
    })
}

fn extract_abstract(doc: &Html) -> String {
    if let Some(desc) = meta_content(doc, "og:description") {
        if desc.len() > 80 {
            return clean_text(&desc);
        }
    }
    for sel_str in [
        "section.pb-8 p",
        "section p",
        "div.prose p",
        "article p",
        "p",
    ] {
        if let Ok(sel) = Selector::parse(sel_str) {
            let paragraphs: Vec<String> = doc
                .select(&sel)
                .map(|el| el.text().collect::<String>().trim().to_string())
                .filter(|s| s.len() > 120)
                .collect();
            if let Some(p) = paragraphs.into_iter().next() {
                return clean_text(&p);
            }
        }
    }
    meta_content(doc, "description").map(|s| clean_text(&s)).unwrap_or_default()
}

fn extract_authors(doc: &Html) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let selectors = [
        "a[href^=\"/papers?authors=\"]",
        "a[href*=\"/author/\"]",
        "a[href*=\"?search=\"]",
        "[data-testid=\"author\"]",
        "span.author",
    ];
    for sel_str in selectors {
        if let Ok(sel) = Selector::parse(sel_str) {
            for el in doc.select(&sel) {
                let raw = clean_text(&el.text().collect::<String>());
                let name = raw
                    .trim_matches(|c: char| c == ',' || c == ';' || c.is_whitespace())
                    .to_string();
                if name.len() < 3 || name.len() > 80 {
                    continue;
                }
                if !name.chars().any(|c| c.is_alphabetic()) {
                    continue;
                }
                if seen.insert(name.clone()) {
                    out.push(name);
                }
                if out.len() >= 20 {
                    break;
                }
            }
        }
        if out.len() >= 20 {
            break;
        }
    }
    out
}

fn extract_upvotes(doc: &Html) -> Option<u32> {
    let upvote_re = Regex::new(r"(?i)upvotes?\s*[:\-]?\s*(\d+)").ok()?;
    let text = doc.root_element().text().collect::<String>();
    upvote_re
        .captures(&text)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse::<u32>().ok())
}

fn extract_submitted_date(doc: &Html) -> String {
    if let Some(t) = meta_content(doc, "article:published_time") {
        return t;
    }
    if let Ok(sel) = Selector::parse("time") {
        if let Some(el) = doc.select(&sel).next() {
            if let Some(dt) = el.value().attr("datetime") {
                return dt.to_string();
            }
            let txt = clean_text(&el.text().collect::<String>());
            if !txt.is_empty() {
                return txt;
            }
        }
    }
    String::new()
}

fn clean_text(s: &str) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.trim().to_string()
}

async fn fetch_with_retry(client: &Client, url: &str, attempts: u32) -> Result<String> {
    let mut delay = std::time::Duration::from_millis(400);
    let mut last: Option<anyhow::Error> = None;
    for attempt in 0..attempts {
        let res = client.get(url).send().await;
        match res {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    return resp
                        .text()
                        .await
                        .with_context(|| format!("read body: {url}"));
                }
                let retriable = status.is_server_error()
                    || status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    || status == reqwest::StatusCode::REQUEST_TIMEOUT;
                last = Some(anyhow!("{url} returned {status}"));
                if !retriable {
                    break;
                }
            }
            Err(e) => {
                last = Some(anyhow!(e).context(format!("fetch: {url}")));
            }
        }
        if attempt + 1 < attempts {
            tokio::time::sleep(delay).await;
            delay = delay.saturating_mul(2);
        }
    }
    Err(last.unwrap_or_else(|| anyhow!("fetch failed: {url}")))
}

fn strip_title_prefix(t: &str) -> String {
    let trimmed = t.trim();
    for prefix in ["Paper page - ", "Paper page – ", "Paper page — ", "Paper page: "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return clean_text(rest);
        }
    }
    clean_text(trimmed)
}
