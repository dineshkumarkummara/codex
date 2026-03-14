use anyhow::Result;
use assert_cmd::Command;
use predicates::str::contains;
use serde_json::json;
use std::path::Path;
use tempfile::TempDir;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::Request;
use wiremock::Respond;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

fn codex_command(codex_home: &Path) -> Result<Command> {
    let mut cmd = Command::new(codex_utils_cargo_bin::cargo_bin("codex")?);
    cmd.env("CODEX_HOME", codex_home);
    cmd.env("OPENAI_API_KEY", "dummy-key");
    Ok(cmd)
}

struct EmbeddingsResponder;

impl Respond for EmbeddingsResponder {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        let body = serde_json::from_slice::<serde_json::Value>(&request.body)
            .unwrap_or_else(|_| json!({ "input": [] }));
        let inputs = body
            .get("input")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();
        let data = inputs
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let text = value.as_str().unwrap_or_default();
                json!({
                    "index": index,
                    "embedding": embedding_for(text),
                })
            })
            .collect::<Vec<_>>();

        ResponseTemplate::new(200).set_body_json(json!({ "data": data }))
    }
}

fn embedding_for(text: &str) -> Vec<f32> {
    let lower = text.to_ascii_lowercase();
    if lower.contains("async image upload") || lower.contains("queue_background_job") {
        return vec![1.0, 0.0];
    }
    if lower.contains("render_button") || lower.contains("render") {
        return vec![0.0, 1.0];
    }
    if lower.contains("fn_symbol_1")
        || lower.contains("fn_symbol_2")
        || lower.contains("var_symbol_")
    {
        return vec![1.0, 0.0];
    }
    if lower.contains("zzq") || lower.contains("yyk") {
        return vec![0.0, 1.0];
    }
    vec![0.5, 0.5]
}

#[tokio::test]
async fn semantic_index_round_trip_uses_mock_embeddings() -> Result<()> {
    let codex_home = TempDir::new()?;
    let repo = TempDir::new()?;
    std::fs::create_dir_all(repo.path().join("src"))?;
    std::fs::write(
        repo.path().join("src/upload.rs"),
        "fn upload_image() {\n    queue_background_job();\n}\n",
    )?;
    std::fs::write(
        repo.path().join("src/ui.rs"),
        "fn render_button() {\n    println!(\"render\");\n}\n",
    )?;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(EmbeddingsResponder)
        .mount(&server)
        .await;

    let mut index_cmd = codex_command(codex_home.path())?;
    index_cmd
        .args([
            "index",
            "--src",
            repo.path().to_string_lossy().as_ref(),
            "--api-base-url",
            &server.uri(),
            "--embedding-model",
            "test-embed",
        ])
        .assert()
        .success()
        .stdout(contains("Indexed 2 chunks"));

    let mut search_cmd = codex_command(codex_home.path())?;
    search_cmd
        .args([
            "search",
            "async image upload",
            "--src",
            repo.path().to_string_lossy().as_ref(),
            "--api-base-url",
            &server.uri(),
            "--top",
            "1",
        ])
        .assert()
        .success()
        .stdout(contains("src/upload.rs:1-3"))
        .stdout(contains("queue_background_job"));

    Ok(())
}

#[tokio::test]
async fn semantic_index_dual_mode_can_recover_from_bad_identifiers() -> Result<()> {
    let codex_home = TempDir::new()?;
    let repo = TempDir::new()?;
    std::fs::create_dir_all(repo.path().join("src"))?;
    std::fs::write(
        repo.path().join("src/weird.rs"),
        "fn zzq(yyk: usize) {\n    let tmp = yyk + 1;\n    println!(\"{tmp}\");\n}\n",
    )?;

    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(EmbeddingsResponder)
        .mount(&server)
        .await;

    let mut index_cmd = codex_command(codex_home.path())?;
    index_cmd
        .args([
            "index",
            "--src",
            repo.path().to_string_lossy().as_ref(),
            "--api-base-url",
            &server.uri(),
            "--normalize-identifiers",
            "--dual-embeddings",
        ])
        .assert()
        .success()
        .stdout(contains("Dual mode"));

    let mut search_cmd = codex_command(codex_home.path())?;
    search_cmd
        .args([
            "search",
            "async image upload",
            "--src",
            repo.path().to_string_lossy().as_ref(),
            "--api-base-url",
            &server.uri(),
            "--top",
            "1",
        ])
        .assert()
        .success()
        .stdout(contains("src/weird.rs:1-4"))
        .stdout(contains("normalized"));

    Ok(())
}
