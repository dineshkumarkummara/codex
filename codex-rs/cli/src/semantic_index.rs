use anyhow::Context;
use anyhow::Result;
use chrono::Utc;
use clap::Parser;
use codex_core::AuthManager;
use codex_core::config::Config;
use codex_core::default_client::build_reqwest_client;
use ignore::WalkBuilder;
use reqwest::StatusCode;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::fs;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use wildmatch::WildMatch;

const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";
const DEFAULT_API_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_INDEX_DIR: &str = ".codex_index";
const DEFAULT_CHUNK_LINES: usize = 200;
const DEFAULT_CHUNK_OVERLAP_LINES: usize = 40;
const DEFAULT_BATCH_SIZE: usize = 32;
const DEFAULT_TOP_K: usize = 8;
const DEFAULT_EXCERPT_LINES: usize = 12;
const DEFAULT_MAX_FILE_BYTES: usize = 256 * 1024;
const INDEX_FORMAT_VERSION: u32 = 2;
const SKIP_DIRECTORIES: &[&str] = &[
    ".codex_index",
    ".git",
    ".hg",
    ".jj",
    ".next",
    ".svn",
    ".turbo",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "vendor",
    "venv",
];

#[derive(Debug, Parser, Clone)]
pub struct IndexCommand {
    /// Directory to index.
    #[arg(long = "src", value_name = "DIR", default_value = ".")]
    pub src: PathBuf,

    /// Directory where the semantic index is stored.
    #[arg(long = "index-dir", value_name = "DIR")]
    pub index_dir: Option<PathBuf>,

    /// Embedding model used for indexing and querying.
    #[arg(long = "embedding-model", value_name = "MODEL", default_value = DEFAULT_EMBEDDING_MODEL)]
    pub embedding_model: String,

    /// Base URL for the embeddings API.
    #[arg(long = "api-base-url", value_name = "URL", default_value = DEFAULT_API_BASE_URL)]
    pub api_base_url: String,

    /// Approximate number of source lines per chunk.
    #[arg(long = "chunk-lines", value_name = "LINES", default_value_t = DEFAULT_CHUNK_LINES)]
    pub chunk_lines: usize,

    /// Number of overlapping lines between adjacent chunks.
    #[arg(
        long = "chunk-overlap-lines",
        value_name = "LINES",
        default_value_t = DEFAULT_CHUNK_OVERLAP_LINES
    )]
    pub chunk_overlap_lines: usize,

    /// Maximum number of chunks to send in each embeddings request.
    #[arg(long = "batch-size", value_name = "COUNT", default_value_t = DEFAULT_BATCH_SIZE)]
    pub batch_size: usize,

    /// Skip files larger than this many bytes.
    #[arg(long = "max-file-bytes", value_name = "BYTES", default_value_t = DEFAULT_MAX_FILE_BYTES)]
    pub max_file_bytes: usize,

    /// Normalize identifiers before embedding to reduce name overfitting.
    #[arg(long = "normalize-identifiers", default_value_t = false)]
    pub normalize_identifiers: bool,

    /// Store both original and normalized embeddings, preferring the better match at search time.
    #[arg(long = "dual-embeddings", default_value_t = false)]
    pub dual_embeddings: bool,
}

#[derive(Debug, Parser, Clone)]
pub struct SearchCommand {
    /// Natural-language query to run against the semantic index.
    #[arg(value_name = "QUERY")]
    pub query: String,

    /// Directory whose semantic index should be searched.
    #[arg(long = "src", value_name = "DIR", default_value = ".")]
    pub src: PathBuf,

    /// Directory where the semantic index is stored.
    #[arg(long = "index-dir", value_name = "DIR")]
    pub index_dir: Option<PathBuf>,

    /// Embedding model to use for the query.
    #[arg(long = "embedding-model", value_name = "MODEL")]
    pub embedding_model: Option<String>,

    /// Base URL for the embeddings API.
    #[arg(long = "api-base-url", value_name = "URL", default_value = DEFAULT_API_BASE_URL)]
    pub api_base_url: String,

    /// Number of matches to display.
    #[arg(long = "top", value_name = "COUNT", default_value_t = DEFAULT_TOP_K)]
    pub top: usize,

    /// Optional glob or extension filter for returned paths.
    #[arg(long = "filter", value_name = "PATTERN")]
    pub filter: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum IndexMode {
    Original,
    Normalized,
    Dual,
}

#[derive(Debug, Serialize, Deserialize)]
struct SemanticIndexFile {
    version: u32,
    root: String,
    embedding_model: String,
    created_at: String,
    chunk_lines: usize,
    chunk_overlap_lines: usize,
    index_mode: IndexMode,
    chunks: Vec<IndexedChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct IndexedChunk {
    path: String,
    start_line: usize,
    end_line: usize,
    sha256: String,
    content: String,
    normalized_content: Option<String>,
    original_embedding: Option<Vec<f32>>,
    normalized_embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ChunkDraft {
    path: String,
    start_line: usize,
    end_line: usize,
    content: String,
    normalized_content: Option<String>,
}

#[derive(Debug, Clone)]
struct SearchHit {
    score: f32,
    original_score: Option<f32>,
    normalized_score: Option<f32>,
    chunk: IndexedChunk,
}

impl Eq for SearchHit {}

impl PartialEq for SearchHit {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score) == Ordering::Equal && self.chunk == other.chunk
    }
}

impl Ord for SearchHit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score)
    }
}

impl PartialOrd for SearchHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub async fn run_index(cmd: IndexCommand, config: &Config) -> Result<()> {
    validate_index_args(&cmd)?;

    let root = canonicalize_existing_dir(&cmd.src)?;
    let index_dir = resolve_index_dir(&root, cmd.index_dir.as_ref());
    let index_mode = resolve_index_mode(&cmd);
    let token = load_auth_token(config).await?;
    let client = EmbeddingsClient::new(cmd.api_base_url, cmd.embedding_model.clone(), token);

    let chunks = collect_chunks(
        &root,
        cmd.chunk_lines,
        cmd.chunk_overlap_lines,
        cmd.max_file_bytes,
        index_mode != IndexMode::Original,
    )?;
    if chunks.is_empty() {
        anyhow::bail!("No text chunks were found under {}.", root.display());
    }

    let chunk_records = embed_chunks(&client, &chunks, index_mode, cmd.batch_size).await?;
    let index = SemanticIndexFile {
        version: INDEX_FORMAT_VERSION,
        root: root.display().to_string(),
        embedding_model: cmd.embedding_model,
        created_at: Utc::now().to_rfc3339(),
        chunk_lines: cmd.chunk_lines,
        chunk_overlap_lines: cmd.chunk_overlap_lines,
        index_mode,
        chunks: chunk_records,
    };

    fs::create_dir_all(&index_dir)
        .with_context(|| format!("failed to create {}", index_dir.display()))?;
    let index_path = index_dir.join("index.json");
    let body = serde_json::to_vec_pretty(&index)?;
    fs::write(&index_path, body)
        .with_context(|| format!("failed to write {}", index_path.display()))?;

    println!(
        "Indexed {} chunks from {} into {} using {} ({:?} mode).",
        index.chunks.len(),
        root.display(),
        index_path.display(),
        index.embedding_model,
        index.index_mode
    );

    Ok(())
}

pub async fn run_search(cmd: SearchCommand, config: &Config) -> Result<()> {
    validate_search_args(&cmd)?;

    let root = canonicalize_existing_dir(&cmd.src)?;
    let index_dir = resolve_index_dir(&root, cmd.index_dir.as_ref());
    let index_path = index_dir.join("index.json");
    let index = read_index(&index_path)?;

    let embedding_model = cmd
        .embedding_model
        .unwrap_or_else(|| index.embedding_model.clone());
    let token = load_auth_token(config).await?;
    let client = EmbeddingsClient::new(cmd.api_base_url, embedding_model.clone(), token);
    let query_embedding = client.embed_query(cmd.query.clone()).await?;

    let hits = rank_chunks(
        &index.chunks,
        &query_embedding,
        index.index_mode,
        cmd.top,
        cmd.filter.as_deref(),
    );
    if hits.is_empty() {
        println!("No semantic matches found in {}.", index_path.display());
        return Ok(());
    }

    println!(
        "Top {} semantic matches from {} using {} ({:?} mode):",
        hits.len(),
        index_path.display(),
        embedding_model,
        index.index_mode
    );
    for (idx, hit) in hits.iter().enumerate() {
        println!(
            "{}. {}:{}-{} (score {:.4}{})",
            idx + 1,
            hit.chunk.path,
            hit.chunk.start_line,
            hit.chunk.end_line,
            hit.score,
            score_breakdown(hit)
        );
        for line in excerpt_lines(&hit.chunk.content, DEFAULT_EXCERPT_LINES) {
            println!("   {line}");
        }
    }

    Ok(())
}

fn validate_index_args(cmd: &IndexCommand) -> Result<()> {
    if cmd.chunk_lines == 0 {
        anyhow::bail!("--chunk-lines must be greater than 0.");
    }
    if cmd.chunk_overlap_lines >= cmd.chunk_lines {
        anyhow::bail!("--chunk-overlap-lines must be smaller than --chunk-lines.");
    }
    if cmd.batch_size == 0 {
        anyhow::bail!("--batch-size must be greater than 0.");
    }
    if cmd.max_file_bytes == 0 {
        anyhow::bail!("--max-file-bytes must be greater than 0.");
    }
    Ok(())
}

fn validate_search_args(cmd: &SearchCommand) -> Result<()> {
    if cmd.query.trim().is_empty() {
        anyhow::bail!("query must not be empty.");
    }
    if cmd.top == 0 {
        anyhow::bail!("--top must be greater than 0.");
    }
    Ok(())
}

fn canonicalize_existing_dir(path: &Path) -> Result<PathBuf> {
    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", path.display()))?;
    if !canonical.is_dir() {
        anyhow::bail!("{} is not a directory.", canonical.display());
    }
    Ok(canonical)
}

fn resolve_index_dir(root: &Path, index_dir: Option<&PathBuf>) -> PathBuf {
    match index_dir {
        Some(path) if path.is_absolute() => path.clone(),
        Some(path) => root.join(path),
        None => root.join(DEFAULT_INDEX_DIR),
    }
}

fn resolve_index_mode(cmd: &IndexCommand) -> IndexMode {
    if cmd.dual_embeddings {
        IndexMode::Dual
    } else if cmd.normalize_identifiers {
        IndexMode::Normalized
    } else {
        IndexMode::Original
    }
}

fn collect_chunks(
    root: &Path,
    chunk_lines: usize,
    chunk_overlap_lines: usize,
    max_file_bytes: usize,
    normalize_identifiers: bool,
) -> Result<Vec<ChunkDraft>> {
    let mut builder = WalkBuilder::new(root);
    builder.hidden(false);
    builder.standard_filters(true);
    builder.git_ignore(true);
    builder.git_global(true);
    builder.git_exclude(true);
    builder.require_git(false);

    let mut chunks = Vec::new();
    for entry in builder.build() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                tracing::warn!("semantic index walk error: {err}");
                continue;
            }
        };
        let path = entry.path();
        if should_skip_path(path, root) {
            continue;
        }
        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::warn!("failed to read {}: {err}", path.display());
                continue;
            }
        };
        if bytes.len() > max_file_bytes || looks_binary(&bytes) {
            continue;
        }

        let Ok(relative_path) = path.strip_prefix(root) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let normalized_text = normalize_identifiers.then(|| normalize_identifier_text(&text));
        chunks.extend(chunk_texts(
            &relative_path.to_string_lossy().replace('\\', "/"),
            &text,
            normalized_text.as_deref(),
            chunk_lines,
            chunk_overlap_lines,
        ));
    }

    Ok(chunks)
}

fn should_skip_path(path: &Path, root: &Path) -> bool {
    let Ok(relative) = path.strip_prefix(root) else {
        return false;
    };
    relative.components().any(|component| {
        let name = component.as_os_str().to_string_lossy();
        SKIP_DIRECTORIES.contains(&name.as_ref())
    })
}

fn looks_binary(bytes: &[u8]) -> bool {
    bytes.contains(&0)
}

fn chunk_texts(
    path: &str,
    text: &str,
    normalized_text: Option<&str>,
    chunk_lines: usize,
    chunk_overlap_lines: usize,
) -> Vec<ChunkDraft> {
    let lines = text.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return Vec::new();
    }

    let normalized_lines = normalized_text.map(|value| value.lines().collect::<Vec<_>>());
    let step = chunk_lines - chunk_overlap_lines;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < lines.len() {
        let end = (start + chunk_lines).min(lines.len());
        let content = lines[start..end].join("\n");
        if !content.trim().is_empty() {
            let normalized_content = normalized_lines
                .as_ref()
                .map(|value| value[start..end].join("\n"))
                .filter(|value| !value.trim().is_empty());
            chunks.push(ChunkDraft {
                path: path.to_string(),
                start_line: start + 1,
                end_line: end,
                content,
                normalized_content,
            });
        }
        if end == lines.len() {
            break;
        }
        start += step;
    }
    chunks
}

fn normalize_identifier_text(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut mappings = HashMap::new();
    let mut next_counts = IdentifierCounters::default();
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut index = 0usize;

    while index < chars.len() {
        let (byte_idx, ch) = chars[index];
        if is_identifier_start(ch) {
            let mut end_idx = index + 1;
            while end_idx < chars.len() && is_identifier_continue(chars[end_idx].1) {
                end_idx += 1;
            }
            let end_byte = if end_idx < chars.len() {
                chars[end_idx].0
            } else {
                text.len()
            };
            let token = &text[byte_idx..end_byte];
            if should_normalize_identifier(token) {
                let next_non_ws = chars[end_idx..]
                    .iter()
                    .map(|(_, value)| *value)
                    .find(|value| !value.is_whitespace());
                let replacement = mappings
                    .entry(token.to_string())
                    .or_insert_with(|| classify_identifier(token, next_non_ws, &mut next_counts))
                    .clone();
                output.push_str(&replacement);
            } else {
                output.push_str(token);
            }
            index = end_idx;
            continue;
        }

        output.push(ch);
        index += 1;
    }

    output
}

fn is_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

fn is_identifier_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

fn should_normalize_identifier(token: &str) -> bool {
    !is_language_keyword(token) && token.chars().any(|ch| ch.is_ascii_alphabetic())
}

fn is_language_keyword(token: &str) -> bool {
    matches!(
        token,
        "abstract"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "case"
            | "catch"
            | "class"
            | "const"
            | "continue"
            | "crate"
            | "def"
            | "default"
            | "del"
            | "do"
            | "elif"
            | "else"
            | "enum"
            | "except"
            | "export"
            | "extends"
            | "false"
            | "final"
            | "finally"
            | "fn"
            | "for"
            | "from"
            | "function"
            | "if"
            | "impl"
            | "import"
            | "in"
            | "interface"
            | "is"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "module"
            | "mut"
            | "new"
            | "nil"
            | "None"
            | "null"
            | "or"
            | "package"
            | "pass"
            | "private"
            | "protected"
            | "pub"
            | "public"
            | "raise"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "switch"
            | "this"
            | "throw"
            | "trait"
            | "true"
            | "try"
            | "type"
            | "typeof"
            | "undefined"
            | "use"
            | "var"
            | "where"
            | "while"
            | "with"
            | "yield"
    )
}

#[derive(Default)]
struct IdentifierCounters {
    fn_symbol: usize,
    type_symbol: usize,
    const_symbol: usize,
    var_symbol: usize,
}

fn classify_identifier(
    token: &str,
    next_non_ws: Option<char>,
    counts: &mut IdentifierCounters,
) -> String {
    if next_non_ws == Some('(') {
        counts.fn_symbol += 1;
        return format!("fn_symbol_{}", counts.fn_symbol);
    }
    if token
        .chars()
        .all(|ch| ch.is_ascii_uppercase() || ch == '_' || ch.is_ascii_digit())
    {
        counts.const_symbol += 1;
        return format!("const_symbol_{}", counts.const_symbol);
    }
    if token
        .chars()
        .next()
        .is_some_and(|ch| ch.is_ascii_uppercase())
    {
        counts.type_symbol += 1;
        return format!("type_symbol_{}", counts.type_symbol);
    }
    counts.var_symbol += 1;
    format!("var_symbol_{}", counts.var_symbol)
}

async fn embed_chunks(
    client: &EmbeddingsClient,
    chunks: &[ChunkDraft],
    index_mode: IndexMode,
    batch_size: usize,
) -> Result<Vec<IndexedChunk>> {
    let mut original_embeddings = Vec::new();
    let mut normalized_embeddings = Vec::new();

    if matches!(index_mode, IndexMode::Original | IndexMode::Dual) {
        for batch in chunks.chunks(batch_size) {
            let input = batch
                .iter()
                .map(|chunk| {
                    embedding_input(
                        &chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        &chunk.content,
                    )
                })
                .collect::<Vec<_>>();
            original_embeddings.extend(client.embed_batch(input).await?);
        }
    }

    if matches!(index_mode, IndexMode::Normalized | IndexMode::Dual) {
        for batch in chunks.chunks(batch_size) {
            let input = batch
                .iter()
                .map(|chunk| {
                    let normalized_content = chunk
                        .normalized_content
                        .as_deref()
                        .unwrap_or(chunk.content.as_str());
                    embedding_input(
                        &chunk.path,
                        chunk.start_line,
                        chunk.end_line,
                        normalized_content,
                    )
                })
                .collect::<Vec<_>>();
            normalized_embeddings.extend(client.embed_batch(input).await?);
        }
    }

    let mut index_chunks = Vec::with_capacity(chunks.len());
    for (idx, chunk) in chunks.iter().enumerate() {
        let original_embedding = match index_mode {
            IndexMode::Original | IndexMode::Dual => Some(
                original_embeddings
                    .get(idx)
                    .cloned()
                    .with_context(|| format!("missing original embedding for chunk {}", idx + 1))?,
            ),
            IndexMode::Normalized => None,
        };
        let normalized_embedding = match index_mode {
            IndexMode::Normalized | IndexMode::Dual => {
                Some(normalized_embeddings.get(idx).cloned().with_context(|| {
                    format!("missing normalized embedding for chunk {}", idx + 1)
                })?)
            }
            IndexMode::Original => None,
        };
        index_chunks.push(IndexedChunk {
            path: chunk.path.clone(),
            start_line: chunk.start_line,
            end_line: chunk.end_line,
            sha256: sha256_hex(&chunk.content),
            content: chunk.content.clone(),
            normalized_content: chunk.normalized_content.clone(),
            original_embedding,
            normalized_embedding,
        });
    }

    Ok(index_chunks)
}

fn embedding_input(path: &str, start_line: usize, end_line: usize, content: &str) -> String {
    format!("path: {path}\nlines: {start_line}-{end_line}\n\n{content}")
}

fn sha256_hex(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn read_index(path: &Path) -> Result<SemanticIndexFile> {
    let file =
        fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let index: SemanticIndexFile = serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse {}", path.display()))?;
    if index.version != INDEX_FORMAT_VERSION {
        anyhow::bail!(
            "Unsupported semantic index format {} in {}.",
            index.version,
            path.display()
        );
    }
    Ok(index)
}

fn rank_chunks(
    chunks: &[IndexedChunk],
    query_embedding: &[f32],
    index_mode: IndexMode,
    top: usize,
    filter: Option<&str>,
) -> Vec<SearchHit> {
    let mut heap = BinaryHeap::new();
    for chunk in chunks {
        if !matches_filter(&chunk.path, filter) {
            continue;
        }

        let original_score = chunk
            .original_embedding
            .as_ref()
            .map(|embedding| cosine_similarity(query_embedding, embedding));
        let normalized_score = chunk
            .normalized_embedding
            .as_ref()
            .map(|embedding| cosine_similarity(query_embedding, embedding));
        let score = match index_mode {
            IndexMode::Original => original_score.unwrap_or(f32::MIN),
            IndexMode::Normalized => normalized_score.unwrap_or(f32::MIN),
            IndexMode::Dual => original_score
                .unwrap_or(f32::MIN)
                .max(normalized_score.unwrap_or(f32::MIN)),
        };
        heap.push(SearchHit {
            score,
            original_score,
            normalized_score,
            chunk: chunk.clone(),
        });
    }

    let mut hits = heap.into_sorted_vec();
    hits.reverse();
    hits.truncate(top);
    hits
}

fn matches_filter(path: &str, filter: Option<&str>) -> bool {
    let Some(filter) = filter.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    if filter.starts_with('.') && !filter.contains('*') && !filter.contains('?') {
        return path.ends_with(filter);
    }
    WildMatch::new(filter).matches(path)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return f32::MIN;
    }
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        dot += lhs * rhs;
        left_norm += lhs * lhs;
        right_norm += rhs * rhs;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return f32::MIN;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn score_breakdown(hit: &SearchHit) -> String {
    match (hit.original_score, hit.normalized_score) {
        (Some(original), Some(normalized)) => format!(
            ", original {:.4}, normalized {:.4}, drift {:.4}",
            original,
            normalized,
            normalized - original
        ),
        (Some(original), None) => format!(", original {original:.4}"),
        (None, Some(normalized)) => format!(", normalized {normalized:.4}"),
        (None, None) => String::new(),
    }
}

fn excerpt_lines(content: &str, max_lines: usize) -> Vec<String> {
    content
        .lines()
        .take(max_lines)
        .map(str::trim_end)
        .map(ToOwned::to_owned)
        .collect()
}

async fn load_auth_token(config: &Config) -> Result<String> {
    if let Ok(api_key) = std::env::var("OPENAI_API_KEY") {
        let api_key = api_key.trim().to_string();
        if !api_key.is_empty() {
            return Ok(api_key);
        }
    }

    let auth_manager = AuthManager::new(
        config.codex_home.clone(),
        true,
        config.cli_auth_credentials_store_mode,
    );
    let Some(auth) = auth_manager.auth().await else {
        anyhow::bail!("No OpenAI credentials found. Run `codex login` or set `OPENAI_API_KEY`.");
    };
    auth.get_token()
        .map_err(anyhow::Error::from)
        .context("failed to load an OpenAI access token")
}

struct EmbeddingsClient {
    http: reqwest::Client,
    api_base_url: String,
    model: String,
    token: String,
}

impl EmbeddingsClient {
    fn new(api_base_url: String, model: String, token: String) -> Self {
        Self {
            http: build_reqwest_client(),
            api_base_url,
            model,
            token,
        }
    }

    async fn embed_query(&self, query: String) -> Result<Vec<f32>> {
        let mut embeddings = self.embed_batch(vec![query]).await?;
        embeddings
            .pop()
            .context("embeddings API did not return a query embedding")
    }

    async fn embed_batch(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingsRequest {
            model: self.model.clone(),
            input,
        };
        let response = self
            .http
            .post(embeddings_url(&self.api_base_url))
            .bearer_auth(&self.token)
            .json(&request)
            .send()
            .await
            .context("failed to call embeddings API")?;

        if response.status() != StatusCode::OK {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable response body>".to_string());
            if status == StatusCode::TOO_MANY_REQUESTS && body.contains("insufficient_quota") {
                anyhow::bail!(
                    "embeddings API returned 429 insufficient_quota; semantic indexing needs available embeddings quota. Full response: {body}"
                );
            }
            anyhow::bail!("embeddings API returned {status}: {body}");
        }

        let mut payload: EmbeddingsResponse = response
            .json()
            .await
            .context("failed to decode embeddings API response")?;
        payload.data.sort_by_key(|item| item.index);
        Ok(payload
            .data
            .into_iter()
            .map(|item| item.embedding)
            .collect())
    }
}

fn embeddings_url(api_base_url: &str) -> String {
    let trimmed = api_base_url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        format!("{trimmed}/embeddings")
    } else {
        format!("{trimmed}/v1/embeddings")
    }
}

#[derive(Debug, Serialize)]
struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingDatum {
    index: usize,
    embedding: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn chunk_text_uses_overlap() {
        let text = (1..=7)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_texts("src/lib.rs", &text, None, 3, 1);
        assert_eq!(
            chunks,
            vec![
                ChunkDraft {
                    path: "src/lib.rs".to_string(),
                    start_line: 1,
                    end_line: 3,
                    content: "line 1\nline 2\nline 3".to_string(),
                    normalized_content: None,
                },
                ChunkDraft {
                    path: "src/lib.rs".to_string(),
                    start_line: 3,
                    end_line: 5,
                    content: "line 3\nline 4\nline 5".to_string(),
                    normalized_content: None,
                },
                ChunkDraft {
                    path: "src/lib.rs".to_string(),
                    start_line: 5,
                    end_line: 7,
                    content: "line 5\nline 6\nline 7".to_string(),
                    normalized_content: None,
                },
            ]
        );
    }

    #[test]
    fn normalize_identifier_text_preserves_keywords() {
        let input = "fn upload_image(ImageJob job) { return process_async(job); }";
        let normalized = normalize_identifier_text(input);
        assert_eq!(
            normalized,
            "fn fn_symbol_1(type_symbol_1 var_symbol_1) { return fn_symbol_2(var_symbol_1); }"
        );
    }

    #[test]
    fn matches_filter_supports_extensions_and_globs() {
        assert!(matches_filter("src/lib.rs", Some(".rs")));
        assert!(!matches_filter("src/lib.ts", Some(".rs")));
        assert!(matches_filter("src/lib.rs", Some("src/*.rs")));
        assert!(!matches_filter("src/lib.rs", Some("tests/*.rs")));
    }

    #[test]
    fn cosine_similarity_handles_mismatch() {
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[1.0]), f32::MIN);
    }

    #[test]
    fn rank_chunks_prefers_normalized_score_in_dual_mode() {
        let chunks = vec![
            IndexedChunk {
                path: "src/alpha.rs".to_string(),
                start_line: 1,
                end_line: 2,
                sha256: "a".to_string(),
                content: "alpha".to_string(),
                normalized_content: Some("fn_symbol_1".to_string()),
                original_embedding: Some(vec![0.2, 0.8]),
                normalized_embedding: Some(vec![1.0, 0.0]),
            },
            IndexedChunk {
                path: "src/beta.rs".to_string(),
                start_line: 1,
                end_line: 2,
                sha256: "b".to_string(),
                content: "beta".to_string(),
                normalized_content: Some("fn_symbol_2".to_string()),
                original_embedding: Some(vec![0.0, 1.0]),
                normalized_embedding: Some(vec![0.0, 1.0]),
            },
        ];

        let hits = rank_chunks(&chunks, &[1.0, 0.0], IndexMode::Dual, 2, None);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk.path, "src/alpha.rs");
        assert!(hits[0].normalized_score.unwrap() > hits[0].original_score.unwrap());
    }

    #[test]
    fn embeddings_url_accepts_both_root_shapes() {
        assert_eq!(
            embeddings_url("https://api.openai.com"),
            "https://api.openai.com/v1/embeddings"
        );
        assert_eq!(
            embeddings_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/embeddings"
        );
    }
}
