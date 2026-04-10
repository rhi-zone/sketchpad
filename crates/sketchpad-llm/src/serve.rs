//! OpenAI-Compatible HTTP Server
//!
//! Provides an HTTP server with OpenAI-compatible API endpoints for
//! chat completions and text completions.
//!
//! This module is only available when the `serve` feature is enabled.
//!
//! # Endpoints
//!
//! - `POST /v1/chat/completions` - Chat completions API
//! - `POST /v1/completions` - Text completions API
//! - `GET /v1/models` - List available models
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

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
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
