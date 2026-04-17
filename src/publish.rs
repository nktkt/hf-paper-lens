use anyhow::{Context, Result, anyhow, bail};
use std::path::Path;
use tokio::process::Command;

pub async fn publish_to_github(
    repo_dir: &Path,
    repo_name: &str,
    description: &str,
) -> Result<String> {
    ensure_git_user(repo_dir).await?;
    run(repo_dir, "git", &["init", "-b", "main"]).await?;
    run(repo_dir, "git", &["add", "-A"]).await?;
    run(
        repo_dir,
        "git",
        &[
            "commit",
            "-m",
            "Initial commit: hf-paper-lens CLI + today's papers snapshot",
        ],
    )
    .await?;

    let url = run_capture(
        repo_dir,
        "gh",
        &[
            "repo",
            "create",
            repo_name,
            "--public",
            "--source",
            ".",
            "--remote",
            "origin",
            "--description",
            description,
            "--push",
        ],
    )
    .await?;
    Ok(url.trim().to_string())
}

async fn ensure_git_user(dir: &Path) -> Result<()> {
    if run_capture(dir, "git", &["config", "--global", "user.name"])
        .await
        .is_err()
    {
        run(
            dir,
            "git",
            &["config", "--global", "user.name", "hf-paper-lens"],
        )
        .await?;
    }
    if run_capture(dir, "git", &["config", "--global", "user.email"])
        .await
        .is_err()
    {
        run(
            dir,
            "git",
            &[
                "config",
                "--global",
                "user.email",
                "hf-paper-lens@users.noreply.github.com",
            ],
        )
        .await?;
    }
    Ok(())
}

async fn run(dir: &Path, bin: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(bin)
        .args(args)
        .current_dir(dir)
        .status()
        .await
        .with_context(|| format!("spawn {bin} {args:?}"))?;
    if !status.success() {
        bail!("{bin} {args:?} exited with status {status}");
    }
    Ok(())
}

async fn run_capture(dir: &Path, bin: &str, args: &[&str]) -> Result<String> {
    let out = Command::new(bin)
        .args(args)
        .current_dir(dir)
        .output()
        .await
        .with_context(|| format!("spawn {bin} {args:?}"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "{bin} {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}
