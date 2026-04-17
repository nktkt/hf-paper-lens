mod codify;
mod fetch;
mod model;
mod prompt;
mod publish;
mod validate;

use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::{Parser, Subcommand};
use futures::stream::{self, StreamExt};
use std::path::PathBuf;

use crate::model::{Paper, Report};

#[derive(Parser, Debug)]
#[command(
    name = "hf-paper-lens",
    about = "Harvest Hugging Face Daily Papers → prompts → codified output → GitHub.",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the full pipeline: fetch → analyze → codify → validate.
    Run {
        /// Target date (YYYY-MM-DD). Defaults to today in UTC.
        #[arg(long)]
        date: Option<String>,
        /// Output directory.
        #[arg(long, default_value = "out")]
        out: PathBuf,
        /// Max concurrent detail fetches.
        #[arg(long, default_value_t = 6)]
        concurrency: usize,
        /// Cap on number of papers (0 = no cap).
        #[arg(long, default_value_t = 0)]
        limit: usize,
    },
    /// Publish a directory as a new public GitHub repo.
    Publish {
        /// Directory that is (or will become) the git repo.
        #[arg(long)]
        repo_dir: PathBuf,
        /// Repository name on GitHub (owner/name or just name).
        #[arg(long)]
        name: String,
        /// Short repository description.
        #[arg(
            long,
            default_value = "Hugging Face Daily Papers harvested, prompted, and codified by hf-paper-lens."
        )]
        description: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run {
            date,
            out,
            concurrency,
            limit,
        } => run_pipeline(date, out, concurrency, limit).await,
        Cmd::Publish {
            repo_dir,
            name,
            description,
        } => {
            let url = publish::publish_to_github(&repo_dir, &name, &description).await?;
            println!("published: {url}");
            Ok(())
        }
    }
}

async fn run_pipeline(
    date: Option<String>,
    out: PathBuf,
    concurrency: usize,
    limit: usize,
) -> Result<()> {
    let target_date = date
        .clone()
        .unwrap_or_else(|| Utc::now().format("%Y-%m-%d").to_string());
    eprintln!("[1/5] fetching paper index for {target_date}");
    let client = fetch::build_client()?;
    let mut ids = fetch::fetch_paper_ids(&client, date.as_deref()).await?;
    ids.sort();
    ids.dedup();
    if limit > 0 && ids.len() > limit {
        ids.truncate(limit);
    }
    eprintln!("  found {} unique papers", ids.len());
    if ids.is_empty() {
        bail!("no papers found for {target_date}");
    }

    eprintln!("[2/5] fetching details (concurrency={concurrency})");
    let concurrency = concurrency.max(1);
    let papers: Vec<Paper> = stream::iter(ids.into_iter())
        .map(|id| {
            let client = client.clone();
            async move {
                match fetch::fetch_paper_detail(&client, &id).await {
                    Ok(p) => Some(p),
                    Err(e) => {
                        eprintln!("  [warn] {id}: {e}");
                        None
                    }
                }
            }
        })
        .buffer_unordered(concurrency)
        .filter_map(|x| async move { x })
        .collect()
        .await;

    let mut papers = papers;
    papers.sort_by(|a, b| a.id.cmp(&b.id));
    papers.dedup_by(|a, b| a.id == b.id);
    eprintln!("  analyzed {} papers", papers.len());

    eprintln!("[3/5] building prompts");
    let prompt_papers: Vec<_> = papers.iter().map(prompt::build_prompt).collect();

    let report = Report {
        generated_at: Utc::now().to_rfc3339(),
        target_date: target_date.clone(),
        total: prompt_papers.len(),
        papers: prompt_papers,
    };

    eprintln!("[4/5] codifying into {}", out.display());
    codify::write_all(&out, &report)
        .await
        .with_context(|| format!("codify into {}", out.display()))?;

    eprintln!("[5/5] validating");
    let vr = validate::enforce(&report)?;
    eprint!("{}", vr.summary());
    eprintln!("ok: {} papers in {}", report.total, out.display());
    Ok(())
}
