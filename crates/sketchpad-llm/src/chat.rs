//! Chat Session Management
//!
//! Provides chat session handling with message history and chat templates
//! for various model formats (Llama, Mistral, ChatML, etc.).
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_llm::{LlmInstance, ChatSession, GenerationConfig};
//!
//! let llm = LlmInstance::load(ModelType::Llama, "./model/", &device)?;
//! let mut session = ChatSession::new(llm, Some("You are a helpful assistant."));
//!
//! let response = session.send("Hello!", &GenerationConfig::default())?;
//! println!("{}", response);
//! ```

use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::inference::{GenerationConfig, LlmError, LlmInstance, ModelType};

/// Role of a chat message
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System prompt that sets the assistant's behavior
    System,
    /// User message
    User,
    /// Assistant (model) response
    Assistant,
}

impl Role {
    /// Get the role as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// A single chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Role of the message sender
    pub role: Role,
    /// Content of the message
    pub content: String,
}

impl ChatMessage {
    /// Create a new chat message
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    /// Create a system message
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}

/// Chat template format for converting messages to prompts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama 2/3 chat format
    Llama,
    /// Mistral/Mixtral format
    Mistral,
    /// ChatML format (used by many models)
    ChatMl,
    /// Phi format
    Phi,
    /// Gemma format
    Gemma,
    /// Qwen format
    Qwen,
    /// Simple format (for models without specific templates)
    Simple,
}

impl ChatTemplate {
    /// Get the appropriate template for a model type
    pub fn for_model(model_type: ModelType) -> Self {
        match model_type {
            ModelType::Llama => Self::Llama,
            ModelType::Mistral | ModelType::Mixtral => Self::Mistral,
            ModelType::Gemma | ModelType::Gemma4 => Self::Gemma,
            ModelType::Phi => Self::Phi,
            ModelType::Qwen => Self::Qwen,
            ModelType::DeepSeek => Self::ChatMl,
            ModelType::Rwkv
            | ModelType::Mamba
            | ModelType::Jamba
            | ModelType::Griffin
            | ModelType::RetNet
            | ModelType::XLstm
            | ModelType::Zamba
            | ModelType::LLaDa
            | ModelType::Tess
            | ModelType::Ttt
            | ModelType::Hyena => Self::Simple,
        }
    }

    /// Format messages into a prompt string
    pub fn format(&self, messages: &[ChatMessage]) -> String {
        match self {
            Self::Llama => format_llama(messages),
            Self::Mistral => format_mistral(messages),
            Self::ChatMl => format_chatml(messages),
            Self::Phi => format_phi(messages),
            Self::Gemma => format_gemma(messages),
            Self::Qwen => format_qwen(messages),
            Self::Simple => format_simple(messages),
        }
    }

    /// Get the stop sequence for this template
    pub fn stop_sequence(&self) -> Option<&'static str> {
        match self {
            Self::Llama => Some("[/INST]"),
            Self::Mistral => Some("[/INST]"),
            Self::ChatMl => Some("<|im_end|>"),
            Self::Phi => Some("<|end|>"),
            Self::Gemma => Some("<end_of_turn>"),
            Self::Qwen => Some("<|im_end|>"),
            Self::Simple => None,
        }
    }
}

/// Llama 2/3 chat format
fn format_llama(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    // Extract system message if present
    let (system_msg, rest) = if !messages.is_empty() && messages[0].role == Role::System {
        (Some(&messages[0].content), &messages[1..])
    } else {
        (None, messages)
    };

    // Format system prompt
    if let Some(system) = system_msg {
        prompt.push_str("<s>[INST] <<SYS>>\n");
        prompt.push_str(system);
        prompt.push_str("\n<</SYS>>\n\n");
    } else {
        prompt.push_str("<s>[INST] ");
    }

    // Format conversation
    let mut first = system_msg.is_none();
    for msg in rest {
        match msg.role {
            Role::User => {
                if !first {
                    prompt.push_str("<s>[INST] ");
                }
                prompt.push_str(&msg.content);
                prompt.push_str(" [/INST]");
                first = false;
            }
            Role::Assistant => {
                prompt.push(' ');
                prompt.push_str(&msg.content);
                prompt.push_str("</s>");
            }
            Role::System => {
                // Skip additional system messages
            }
        }
    }

    prompt
}

/// Mistral chat format
fn format_mistral(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role {
            Role::System | Role::User => {
                prompt.push_str("[INST] ");
                prompt.push_str(&msg.content);
                prompt.push_str(" [/INST]");
            }
            Role::Assistant => {
                prompt.push_str(&msg.content);
                prompt.push_str("</s>");
            }
        }
    }

    prompt
}

/// ChatML format
fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        prompt.push_str("<|im_start|>");
        prompt.push_str(msg.role.as_str());
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("<|im_end|>\n");
    }

    // Add start of assistant turn
    prompt.push_str("<|im_start|>assistant\n");

    prompt
}

/// Phi format
fn format_phi(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        prompt.push_str("<|");
        prompt.push_str(msg.role.as_str());
        prompt.push_str("|>\n");
        prompt.push_str(&msg.content);
        prompt.push_str("<|end|>\n");
    }

    prompt.push_str("<|assistant|>\n");

    prompt
}

/// Gemma format
fn format_gemma(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        let role = match msg.role {
            Role::System | Role::User => "user",
            Role::Assistant => "model",
        };
        prompt.push_str("<start_of_turn>");
        prompt.push_str(role);
        prompt.push('\n');
        prompt.push_str(&msg.content);
        prompt.push_str("<end_of_turn>\n");
    }

    prompt.push_str("<start_of_turn>model\n");

    prompt
}

/// Qwen format (ChatML variant)
fn format_qwen(messages: &[ChatMessage]) -> String {
    // Qwen uses ChatML format
    format_chatml(messages)
}

/// Simple format for models without specific templates
fn format_simple(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                prompt.push_str("System: ");
                prompt.push_str(&msg.content);
                prompt.push_str("\n\n");
            }
            Role::User => {
                prompt.push_str("User: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
            Role::Assistant => {
                prompt.push_str("Assistant: ");
                prompt.push_str(&msg.content);
                prompt.push('\n');
            }
        }
    }

    prompt.push_str("Assistant: ");

    prompt
}

/// Chat session that maintains message history
pub struct ChatSession<B: Backend> {
    /// The underlying LLM instance
    llm: LlmInstance<B>,
    /// Message history
    messages: Vec<ChatMessage>,
    /// Chat template format
    template: ChatTemplate,
}

impl<B: Backend> ChatSession<B> {
    /// Create a new chat session
    ///
    /// # Arguments
    ///
    /// * `llm` - The LLM instance to use for generation
    /// * `system_prompt` - Optional system prompt to set the assistant's behavior
    pub fn new(llm: LlmInstance<B>, system_prompt: Option<&str>) -> Self {
        let template = ChatTemplate::for_model(llm.model_type());
        let mut messages = Vec::new();

        if let Some(system) = system_prompt {
            messages.push(ChatMessage::system(system));
        }

        Self {
            llm,
            messages,
            template,
        }
    }

    /// Create a new chat session with a specific template
    pub fn with_template(
        llm: LlmInstance<B>,
        system_prompt: Option<&str>,
        template: ChatTemplate,
    ) -> Self {
        let mut messages = Vec::new();

        if let Some(system) = system_prompt {
            messages.push(ChatMessage::system(system));
        }

        Self {
            llm,
            messages,
            template,
        }
    }

    /// Send a user message and get the assistant's response
    ///
    /// # Arguments
    ///
    /// * `message` - The user's message
    /// * `config` - Generation configuration
    ///
    /// # Returns
    ///
    /// The assistant's response text
    pub fn send(&mut self, message: &str, config: &GenerationConfig) -> Result<String, LlmError> {
        // Add user message to history
        self.messages.push(ChatMessage::user(message));

        // Format full prompt
        let prompt = self.template.format(&self.messages);

        // Build generation config with template stop sequence
        let mut gen_config = config.clone();
        if let Some(stop) = self.template.stop_sequence() {
            if !gen_config.stop_sequences.contains(&stop.to_string()) {
                gen_config.stop_sequences.push(stop.to_string());
            }
        }

        // Generate response
        let response = self.llm.generate(&prompt, &gen_config)?;

        // Add assistant response to history
        self.messages.push(ChatMessage::assistant(&response));

        Ok(response)
    }

    /// Get the current message history
    pub fn messages(&self) -> &[ChatMessage] {
        &self.messages
    }

    /// Clear the message history (keeps system prompt if any)
    pub fn clear(&mut self) {
        let system = self
            .messages
            .first()
            .filter(|m| m.role == Role::System)
            .cloned();
        self.messages.clear();
        if let Some(msg) = system {
            self.messages.push(msg);
        }
    }

    /// Get a reference to the underlying LLM
    pub fn llm(&self) -> &LlmInstance<B> {
        &self.llm
    }

    /// Get the chat template being used
    pub fn template(&self) -> ChatTemplate {
        self.template
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("Be helpful");
        assert_eq!(sys.role, Role::System);
        assert_eq!(sys.content, "Be helpful");

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, Role::User);

        let asst = ChatMessage::assistant("Hi there!");
        assert_eq!(asst.role, Role::Assistant);
    }

    #[test]
    fn test_chatml_format() {
        let messages = vec![
            ChatMessage::system("You are helpful."),
            ChatMessage::user("Hello"),
        ];

        let prompt = format_chatml(&messages);
        assert!(prompt.contains("<|im_start|>system"));
        assert!(prompt.contains("<|im_start|>user"));
        assert!(prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_llama_format() {
        let messages = vec![ChatMessage::system("Be concise."), ChatMessage::user("Hi")];

        let prompt = format_llama(&messages);
        assert!(prompt.contains("<<SYS>>"));
        assert!(prompt.contains("[INST]"));
    }

    #[test]
    fn test_simple_format() {
        let messages = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi!"),
            ChatMessage::user("How are you?"),
        ];

        let prompt = format_simple(&messages);
        assert!(prompt.contains("User: Hello"));
        assert!(prompt.contains("Assistant: Hi!"));
        assert!(prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_template_for_model() {
        assert_eq!(
            ChatTemplate::for_model(ModelType::Llama),
            ChatTemplate::Llama
        );
        assert_eq!(
            ChatTemplate::for_model(ModelType::Mistral),
            ChatTemplate::Mistral
        );
        assert_eq!(
            ChatTemplate::for_model(ModelType::Mamba),
            ChatTemplate::Simple
        );
    }
}
