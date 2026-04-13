//! Tokenizer loading from GGUF metadata
//!
//! GGUF files embed the full tokenizer vocabulary under `tokenizer.ggml.*` keys
//! and the chat template under `tokenizer.chat_template`.
//! This module reconstructs a `tokenizers::Tokenizer` directly from that data,
//! removing the need for a separate tokenizer.json file.

use std::collections::HashMap;

use ahash::AHashMap;
use minijinja::Environment;
use sketchpad_convert::gguf::{GgufFile, MetadataValue};
use tokenizers::{
    AddedToken, Tokenizer,
    decoders::metaspace::Metaspace as MetaspaceDecoder,
    models::bpe::BPE,
    pre_tokenizers::metaspace::{Metaspace, PrependScheme},
};

/// Error returned when GGUF tokenizer data is missing or malformed
#[derive(Debug)]
pub struct GgufTokenizerError(pub String);

impl std::fmt::Display for GgufTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GGUF tokenizer error: {}", self.0)
    }
}

impl std::error::Error for GgufTokenizerError {}

/// Load a `tokenizers::Tokenizer` from the embedded vocabulary in a GGUF file.
///
/// Supports `tokenizer.ggml.model = "llama"` (SentencePiece-style BPE with byte fallback),
/// which is used by Gemma 4 and most modern models stored in GGUF format.
pub fn load_tokenizer(file: &GgufFile) -> Result<Tokenizer, GgufTokenizerError> {
    let meta = file.metadata();

    let tokens = string_array(meta, "tokenizer.ggml.tokens")
        .ok_or_else(|| GgufTokenizerError("missing tokenizer.ggml.tokens".into()))?;

    let token_types =
        u32_array(meta, "tokenizer.ggml.token_type").unwrap_or_else(|| vec![0u32; tokens.len()]);

    let merge_strs = string_array(meta, "tokenizer.ggml.merges").unwrap_or_default();

    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    let merges: Vec<(String, String)> = merge_strs
        .iter()
        .filter_map(|m| {
            let mut it = m.splitn(2, ' ');
            let a = it.next()?.to_string();
            let b = it.next()?.to_string();
            Some((a, b))
        })
        .collect();

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .byte_fallback(true)
        .build()
        .map_err(|e| GgufTokenizerError(format!("BPE build failed: {e}")))?;

    let mut tokenizer = Tokenizer::new(bpe);

    tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
    tokenizer.with_decoder(Some(MetaspaceDecoder::new(
        '▁',
        PrependScheme::Always,
        true,
    )));

    // Register control/special tokens so they are never BPE-split
    let special: Vec<AddedToken> = tokens
        .iter()
        .zip(token_types.iter())
        .filter(|&(_, &tt)| tt == 3)
        .map(|(text, _)| AddedToken::from(text.as_str(), true))
        .collect();
    tokenizer.add_special_tokens(&special);

    Ok(tokenizer)
}

/// A chat message: role (`"user"`, `"model"`, `"system"`) and content.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
        }
    }

    pub fn model(content: impl Into<String>) -> Self {
        Self {
            role: "model".into(),
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
        }
    }
}

/// Apply the model's chat template (read from GGUF) to format a conversation.
///
/// The template is evaluated with [minijinja] using the same context that
/// HuggingFace's `apply_chat_template` provides: `messages`, `bos_token`,
/// `eos_token`, and `add_generation_prompt`.
///
/// Call with `add_generation_prompt = true` to append the model turn opener,
/// making the output ready for the model to continue.
pub fn apply_chat_template(
    file: &GgufFile,
    messages: &[ChatMessage],
    add_generation_prompt: bool,
) -> Result<String, GgufTokenizerError> {
    let template_str = raw_chat_template(file)
        .ok_or_else(|| GgufTokenizerError("no chat_template in GGUF metadata".into()))?;

    // Resolve BOS/EOS token strings from the vocabulary
    let tokens = string_array(file.metadata(), "tokenizer.ggml.tokens").unwrap_or_default();
    let bos_id = scalar_u32(file.metadata(), "tokenizer.ggml.bos_token_id").unwrap_or(2) as usize;
    let eos_id = scalar_u32(file.metadata(), "tokenizer.ggml.eos_token_id").unwrap_or(1) as usize;
    let bos_token = tokens
        .get(bos_id)
        .cloned()
        .unwrap_or_else(|| "<bos>".into());
    let eos_token = tokens
        .get(eos_id)
        .cloned()
        .unwrap_or_else(|| "<eos>".into());

    let messages_val = minijinja::Value::from_serialize(messages);

    let mut env = Environment::new();
    minijinja_contrib::add_to_environment(&mut env);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", &template_str)
        .map_err(|e| GgufTokenizerError(format!("chat template parse error: {e}")))?;

    env.get_template("chat")
        .unwrap()
        .render(minijinja::context! {
            messages => messages_val,
            bos_token => bos_token,
            eos_token => eos_token,
            add_generation_prompt => add_generation_prompt,
        })
        .map_err(|e| GgufTokenizerError(format!("chat template render error: {e}")))
}

/// Return the raw Jinja2 chat template string from GGUF metadata, if present.
pub fn raw_chat_template(file: &GgufFile) -> Option<String> {
    match file.metadata().get("tokenizer.chat_template") {
        Some(MetadataValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

// --- helpers ---

fn string_array(meta: &HashMap<String, MetadataValue>, key: &str) -> Option<Vec<String>> {
    match meta.get(key)? {
        MetadataValue::Array(arr) => Some(
            arr.iter()
                .map(|v| match v {
                    MetadataValue::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => None,
    }
}

fn u32_array(meta: &HashMap<String, MetadataValue>, key: &str) -> Option<Vec<u32>> {
    match meta.get(key)? {
        MetadataValue::Array(arr) => Some(
            arr.iter()
                .map(|v| match v {
                    MetadataValue::U32(n) => *n,
                    MetadataValue::I32(n) => *n as u32,
                    MetadataValue::U8(n) => *n as u32,
                    _ => 0,
                })
                .collect(),
        ),
        _ => None,
    }
}

fn scalar_u32(meta: &HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    match meta.get(key)? {
        MetadataValue::U32(n) => Some(*n),
        MetadataValue::I32(n) => Some(*n as u32),
        MetadataValue::U64(n) => Some(*n as u32),
        MetadataValue::U8(n) => Some(*n as u32),
        _ => None,
    }
}
