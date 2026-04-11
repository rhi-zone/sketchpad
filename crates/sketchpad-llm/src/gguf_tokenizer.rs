//! Tokenizer loading from GGUF metadata
//!
//! GGUF files embed the full tokenizer vocabulary under `tokenizer.ggml.*` keys
//! and the chat template under `tokenizer.chat_template`.
//! This module reconstructs a `tokenizers::Tokenizer` directly from that data,
//! removing the need for a separate tokenizer.json file.

use std::collections::HashMap;

use ahash::AHashMap;

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

    // Token strings — one per vocabulary entry, indexed by token ID
    let tokens = string_array(meta, "tokenizer.ggml.tokens")
        .ok_or_else(|| GgufTokenizerError("missing tokenizer.ggml.tokens".into()))?;

    // Token types: 0=normal, 1=unknown, 2=unused, 3=control/special, 6=byte
    let token_types =
        u32_array(meta, "tokenizer.ggml.token_type").unwrap_or_else(|| vec![0u32; tokens.len()]);

    // BPE merge rules (may be absent for pure unigram models, but present for Gemma)
    let merge_strs = string_array(meta, "tokenizer.ggml.merges").unwrap_or_default();

    // Build vocab map: token string → id (AHashMap required by tokenizers crate)
    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    // Parse merge rules: "A B" → ("A", "B")
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

    // SentencePiece-style tokenizers use a leading `▁` to mark word boundaries.
    // `PrependScheme::Always` prepends it before the very first token of a sequence.
    tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
    tokenizer.with_decoder(Some(MetaspaceDecoder::new(
        '▁',
        PrependScheme::Always,
        true,
    )));

    // Register control/special tokens (token_type == 3) as AddedTokens so the
    // tokenizer never tries to BPE-split them.
    let special: Vec<AddedToken> = tokens
        .iter()
        .zip(token_types.iter())
        .filter(|&(_, &tt)| tt == 3)
        .map(|(text, _)| AddedToken::from(text.as_str(), true))
        .collect();
    tokenizer.add_special_tokens(&special);

    Ok(tokenizer)
}

/// Read the Jinja2 chat template stored in GGUF metadata, if present.
pub fn chat_template(file: &GgufFile) -> Option<String> {
    match file.metadata().get("tokenizer.chat_template") {
        Some(MetadataValue::String(s)) => Some(s.clone()),
        _ => None,
    }
}

/// Format a single-turn user prompt in Gemma IT chat format.
///
/// Produces: `<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n`
///
/// For multi-turn or system-prompt scenarios, use [`format_gemma_chat`].
pub fn format_gemma_prompt(prompt: &str) -> String {
    format!("<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
}

/// Format a multi-turn conversation in Gemma IT chat format.
///
/// Each message is `(role, content)` where role is `"user"`, `"model"`, or `"system"`.
/// A generation prompt (`<start_of_turn>model\n`) is appended if `add_generation_prompt` is true.
pub fn format_gemma_chat(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut out = String::from("<bos>");
    for (role, content) in messages {
        out.push_str(&format!("<start_of_turn>{role}\n{content}<end_of_turn>\n"));
    }
    if add_generation_prompt {
        out.push_str("<start_of_turn>model\n");
    }
    out
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
