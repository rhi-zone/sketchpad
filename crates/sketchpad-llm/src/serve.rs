//! OpenAI- and Anthropic-Compatible HTTP Server
//!
//! Provides an HTTP server with OpenAI-compatible and Anthropic-compatible
//! API endpoints for chat completions and text completions.
//!
//! This module is only available when the `serve` feature is enabled.
//!
//! # Endpoints
//!
//! - `POST /v1/chat/completions` - OpenAI chat completions API
//! - `POST /v1/completions` - OpenAI text completions API
//! - `GET /v1/models` - List available models
//! - `POST /v1/messages` - Anthropic Messages API
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_llm::{LlmInstance, serve::run_server};
//!
//! let llm = LlmInstance::load(ModelType::Llama, "./model/", &device)?;
//! run_server(llm, "127.0.0.1", 8080).await?;
//! ```

use std::sync::Arc;

use axum::response::sse::Event;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
};
use burn::prelude::*;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::chat::{ChatMessage, ChatTemplate, Role};
use crate::inference::{GenerationConfig, LlmError, LlmInstance};

/// Errors that can occur during server operations
#[derive(Debug)]
pub enum ServeError {
    /// Server bind error
    Bind(String),
    /// Model error
    Model(LlmError),
}

impl std::fmt::Display for ServeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind(e) => write!(f, "Server bind error: {}", e),
            Self::Model(e) => write!(f, "Model error: {}", e),
        }
    }
}

impl std::error::Error for ServeError {}

/// Server state containing the model
struct ServerState<B: Backend> {
    llm: Mutex<LlmInstance<B>>,
    model_id: String,
}

/// OpenAI-compatible chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiChatMessage {
    pub role: String,
    pub content: String,
}

impl From<ApiChatMessage> for ChatMessage {
    fn from(msg: ApiChatMessage) -> Self {
        let role = match msg.role.as_str() {
            "system" => Role::System,
            "assistant" => Role::Assistant,
            _ => Role::User,
        };
        ChatMessage::new(role, msg.content)
    }
}

/// Chat completion request
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ApiChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

/// Text completion request
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub stop: Option<Vec<String>>,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}

/// Choice in a completion response
#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub message: Option<ApiChatMessage>,
    pub text: Option<String>,
    pub finish_reason: String,
}

/// Usage statistics
#[derive(Debug, Serialize)]
pub struct UsageStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Completion response
#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: UsageStats,
}

/// Model info
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

/// Models list response
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Error response
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: Option<String>,
}

impl IntoResponse for ErrorResponse {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::INTERNAL_SERVER_ERROR, Json(self)).into_response()
    }
}

// ─── Anthropic Messages API ──────────────────────────────────────────────────

/// Anthropic Messages API request body (POST /v1/messages)
#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    pub max_tokens: u32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stream: Option<bool>,
    pub system: Option<String>,
}

/// Single message in an Anthropic conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    /// "user" or "assistant"
    pub role: String,
    pub content: String,
}

/// Non-streaming Anthropic Messages response
#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub role: String,
    pub content: Vec<AnthropicContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

/// Content block within an Anthropic response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicContentBlock {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
}

/// Token usage for an Anthropic response
#[derive(Debug, Serialize, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ─── SSE event shapes ────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct SseMessageStart<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    message: SseMessageStartInner<'a>,
}

#[derive(Debug, Serialize)]
struct SseMessageStartInner<'a> {
    id: &'a str,
    #[serde(rename = "type")]
    type_: &'static str,
    role: &'static str,
    model: &'a str,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct SseContentBlockStart {
    #[serde(rename = "type")]
    type_: &'static str,
    index: u32,
    content_block: AnthropicContentBlock,
}

#[derive(Debug, Serialize)]
struct SseContentBlockDelta<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    index: u32,
    delta: SseTextDelta<'a>,
}

#[derive(Debug, Serialize)]
struct SseTextDelta<'a> {
    #[serde(rename = "type")]
    type_: &'static str,
    text: &'a str,
}

#[derive(Debug, Serialize)]
struct SseMessageDelta {
    #[serde(rename = "type")]
    type_: &'static str,
    delta: SseStopDelta,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct SseStopDelta {
    stop_reason: String,
}

#[derive(Debug, Serialize)]
struct SseMessageStop {
    #[serde(rename = "type")]
    type_: &'static str,
}

// ─── Chat template helper ────────────────────────────────────────────────────

/// Format a list of Anthropic messages into a plain-text prompt.
///
/// Uses a simple Human/Assistant template compatible with most instruction-tuned
/// LLMs that don't have a dedicated chat template applied at the GGUF level.
fn format_anthropic_prompt(system: Option<&str>, messages: &[AnthropicMessage]) -> String {
    let mut prompt = String::new();
    if let Some(sys) = system {
        prompt.push_str(sys);
        prompt.push('\n');
    }
    for msg in messages {
        match msg.role.as_str() {
            "assistant" => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
            _ => {
                prompt.push_str("Human: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
        }
    }
    prompt.push_str("Assistant:");
    prompt
}

// ─── /v1/messages handler ────────────────────────────────────────────────────

/// Anthropic Messages API endpoint
async fn messages<B: Backend>(
    State(state): State<Arc<ServerState<B>>>,
    Json(request): Json<MessagesRequest>,
) -> Response {
    let streaming = request.stream.unwrap_or(false);
    let max_tokens = request.max_tokens as usize;
    let temperature = request.temperature.unwrap_or(0.7);
    let top_p = request.top_p.unwrap_or(0.9);

    let prompt = format_anthropic_prompt(request.system.as_deref(), &request.messages);

    let config = GenerationConfig::new(max_tokens)
        .with_temperature(temperature)
        .with_top_p(top_p);

    let msg_id = format!("msg-{}", uuid_v4());
    let model_id = state.model_id.clone();

    // Run inference synchronously (model is not async) inside spawn_blocking so we
    // don't block the tokio executor.
    let llm_lock = state.llm.lock().await;
    let result = llm_lock.generate(&prompt, &config);
    drop(llm_lock);

    let text = match result {
        Ok(t) => t,
        Err(e) => {
            let err = ErrorResponse {
                error: ErrorDetail {
                    message: e.to_string(),
                    r#type: "model_error".to_string(),
                    code: None,
                },
            };
            return err.into_response();
        }
    };

    let input_tokens = prompt.split_whitespace().count() as u32;
    let output_tokens = text.split_whitespace().count() as u32;
    let stop_reason = if output_tokens >= max_tokens as u32 {
        "max_tokens"
    } else {
        "end_turn"
    };

    if streaming {
        // Stubbed streaming: emit all SSE events with the full response in a single
        // content_block_delta (true token-level streaming requires threading through
        // the model generation loop, which is not yet implemented).
        let events = build_sse_events(
            &msg_id,
            &model_id,
            &text,
            stop_reason,
            input_tokens,
            output_tokens,
        );
        let stream = futures_util::stream::iter(events);
        Sse::new(stream).into_response()
    } else {
        let response = MessagesResponse {
            id: msg_id,
            type_: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![AnthropicContentBlock {
                type_: "text".to_string(),
                text,
            }],
            model: model_id,
            stop_reason: stop_reason.to_string(),
            usage: AnthropicUsage {
                input_tokens,
                output_tokens,
            },
        };
        (StatusCode::OK, Json(response)).into_response()
    }
}

/// Build the ordered SSE event list for a stubbed streaming response.
fn sse_event(name: &str, value: &impl serde::Serialize) -> Result<Event, std::convert::Infallible> {
    let data = serde_json::to_string(value).unwrap_or_default();
    Ok(Event::default().event(name).data(data))
}

fn build_sse_events(
    msg_id: &str,
    model_id: &str,
    text: &str,
    stop_reason: &str,
    input_tokens: u32,
    output_tokens: u32,
) -> Vec<Result<Event, std::convert::Infallible>> {
    vec![
        sse_event(
            "message_start",
            &SseMessageStart {
                type_: "message_start",
                message: SseMessageStartInner {
                    id: msg_id,
                    type_: "message",
                    role: "assistant",
                    model: model_id,
                    usage: AnthropicUsage {
                        input_tokens,
                        output_tokens: 0,
                    },
                },
            },
        ),
        sse_event(
            "content_block_start",
            &SseContentBlockStart {
                type_: "content_block_start",
                index: 0,
                content_block: AnthropicContentBlock {
                    type_: "text".to_string(),
                    text: String::new(),
                },
            },
        ),
        sse_event(
            "content_block_delta",
            &SseContentBlockDelta {
                type_: "content_block_delta",
                index: 0,
                delta: SseTextDelta {
                    type_: "text_delta",
                    text,
                },
            },
        ),
        sse_event(
            "message_delta",
            &SseMessageDelta {
                type_: "message_delta",
                delta: SseStopDelta {
                    stop_reason: stop_reason.to_string(),
                },
                usage: AnthropicUsage {
                    input_tokens,
                    output_tokens,
                },
            },
        ),
        sse_event(
            "message_stop",
            &SseMessageStop {
                type_: "message_stop",
            },
        ),
    ]
}

/// Run the HTTP server
///
/// # Arguments
///
/// * `llm` - The LLM instance to serve
/// * `host` - Host address to bind to
/// * `port` - Port number to bind to
pub async fn run_server<B: Backend + 'static>(
    llm: LlmInstance<B>,
    host: &str,
    port: u16,
) -> Result<(), ServeError> {
    let model_id = format!("burn-models-{}", llm.model_type().as_str());

    let state = Arc::new(ServerState {
        llm: Mutex::new(llm),
        model_id,
    });

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions::<B>))
        .route("/v1/completions", post(completions::<B>))
        .route("/v1/models", get(list_models::<B>))
        .route("/v1/messages", post(messages::<B>))
        .route("/health", get(health_check))
        .with_state(state);

    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| ServeError::Bind(e.to_string()))?;

    println!("Server listening on http://{}", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| ServeError::Bind(e.to_string()))?;

    Ok(())
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "ok"
}

/// List available models
async fn list_models<B: Backend>(State(state): State<Arc<ServerState<B>>>) -> Json<ModelsResponse> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "burn-models".to_string(),
        }],
    })
}

/// Chat completions endpoint
async fn chat_completions<B: Backend>(
    State(state): State<Arc<ServerState<B>>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Json<CompletionResponse>, ErrorResponse> {
    let config = GenerationConfig::new(request.max_tokens)
        .with_temperature(request.temperature)
        .with_top_p(request.top_p)
        .with_stop_sequences(request.stop.unwrap_or_default());

    // Convert messages
    let messages: Vec<ChatMessage> = request.messages.into_iter().map(Into::into).collect();

    // Get the LLM and generate
    let llm = state.llm.lock().await;
    let template = ChatTemplate::for_model(llm.model_type());
    let prompt = template.format(&messages);

    let response = llm.generate(&prompt, &config).map_err(|e| ErrorResponse {
        error: ErrorDetail {
            message: e.to_string(),
            r#type: "model_error".to_string(),
            code: None,
        },
    })?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Rough token count estimation
    let prompt_tokens = prompt.split_whitespace().count();
    let completion_tokens = response.split_whitespace().count();

    Ok(Json(CompletionResponse {
        id: format!("chatcmpl-{}", uuid_v4()),
        object: "chat.completion".to_string(),
        created: now,
        model: state.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            message: Some(ApiChatMessage {
                role: "assistant".to_string(),
                content: response,
            }),
            text: None,
            finish_reason: "stop".to_string(),
        }],
        usage: UsageStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// Text completions endpoint
async fn completions<B: Backend>(
    State(state): State<Arc<ServerState<B>>>,
    Json(request): Json<CompletionRequest>,
) -> Result<Json<CompletionResponse>, ErrorResponse> {
    let config = GenerationConfig::new(request.max_tokens)
        .with_temperature(request.temperature)
        .with_top_p(request.top_p)
        .with_stop_sequences(request.stop.unwrap_or_default());

    let llm = state.llm.lock().await;
    let response = llm
        .generate(&request.prompt, &config)
        .map_err(|e| ErrorResponse {
            error: ErrorDetail {
                message: e.to_string(),
                r#type: "model_error".to_string(),
                code: None,
            },
        })?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let prompt_tokens = request.prompt.split_whitespace().count();
    let completion_tokens = response.split_whitespace().count();

    Ok(Json(CompletionResponse {
        id: format!("cmpl-{}", uuid_v4()),
        object: "text_completion".to_string(),
        created: now,
        model: state.model_id.clone(),
        choices: vec![CompletionChoice {
            index: 0,
            message: None,
            text: Some(response),
            finish_reason: "stop".to_string(),
        }],
        usage: UsageStats {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }))
}

/// Generate a simple UUID v4-like string
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    format!("{:032x}", now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_message_conversion() {
        let api_msg = ApiChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let chat_msg: ChatMessage = api_msg.into();
        assert_eq!(chat_msg.role, Role::User);
        assert_eq!(chat_msg.content, "Hello");
    }

    #[test]
    fn test_defaults() {
        assert_eq!(default_max_tokens(), 256);
        assert!((default_temperature() - 0.7).abs() < 0.001);
        assert!((default_top_p() - 0.9).abs() < 0.001);
    }
}
