//! CLIP BPE Tokenizer
//!
//! Implements the Byte Pair Encoding tokenizer used by CLIP models.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use regex::Regex;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid vocabulary file format")]
    InvalidVocab,

    #[error("Regex error: {0}")]
    Regex(#[from] regex::Error),
}

/// Special token IDs
pub const START_OF_TEXT: u32 = 49406;
pub const END_OF_TEXT: u32 = 49407;

/// Embedded CLIP BPE merges (from OpenAI's CLIP)
const CLIP_BPE_MERGES: &str = include_str!("../data/bpe_merges.txt");

/// Embedded CLIP vocabulary (from OpenAI's CLIP)
const CLIP_VOCAB_JSON: &str = include_str!("../data/vocab.json");

/// CLIP BPE Tokenizer
pub struct ClipTokenizer {
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    cache: std::cell::RefCell<HashMap<String, String>>,
    pat: Regex,
}

impl ClipTokenizer {
    /// Create a new tokenizer with the standard CLIP vocabulary (embedded)
    ///
    /// This uses the same vocabulary and BPE merges as OpenAI's CLIP model,
    /// which is standard for all Stable Diffusion 1.x and 2.x models.
    pub fn new() -> Self {
        Self::from_vocab_and_merges(CLIP_VOCAB_JSON, CLIP_BPE_MERGES)
            .expect("embedded CLIP vocabulary should be valid")
    }

    /// Create a new tokenizer from vocabulary and merges files
    pub fn from_file<P: AsRef<Path>>(vocab_path: P) -> Result<Self, TokenizerError> {
        let vocab_content = fs::read_to_string(&vocab_path)?;

        // Check if this is a vocab.json or bpe_merges.txt
        if vocab_path.as_ref().extension().is_some_and(|e| e == "json") {
            // Load vocab.json, use embedded merges
            Self::from_vocab_and_merges(&vocab_content, CLIP_BPE_MERGES)
        } else {
            // Load as merges file, build vocab from merges (legacy path)
            Self::from_merges_only(&vocab_content)
        }
    }

    /// Create tokenizer from vocab.json and bpe_merges.txt content
    fn from_vocab_and_merges(vocab_json: &str, merges: &str) -> Result<Self, TokenizerError> {
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect();

        // Parse vocab.json (simple format: {"token": id, ...})
        let encoder = parse_vocab_json(vocab_json)?;
        let decoder: HashMap<u32, String> = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        // Parse BPE merges
        let bpe_ranks = parse_bpe_merges(merges)?;

        // Regex pattern for tokenization (matches CLIP's pattern)
        let pat = Regex::new(
            r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )?;

        Ok(Self {
            byte_encoder,
            byte_decoder,
            encoder,
            decoder,
            bpe_ranks,
            cache: std::cell::RefCell::new(HashMap::new()),
            pat,
        })
    }

    /// Create tokenizer from merges only (builds vocab programmatically)
    ///
    /// This is the legacy path - prefer using vocab.json for correctness.
    fn from_merges_only(merges: &str) -> Result<Self, TokenizerError> {
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect();

        let bpe_ranks = parse_bpe_merges(merges)?;

        // Build encoder vocabulary from merges
        // IMPORTANT: Must iterate in deterministic order to get consistent token IDs
        let mut encoder = HashMap::new();
        let mut vocab_idx = 0u32;

        // Add byte-level tokens (sorted for deterministic order)
        let mut byte_chars: Vec<_> = byte_encoder.values().collect();
        byte_chars.sort();
        for c in &byte_chars {
            encoder.insert(c.to_string(), vocab_idx);
            vocab_idx += 1;
        }

        // Add byte-level tokens with end-of-word marker
        for c in &byte_chars {
            encoder.insert(format!("{}</w>", c), vocab_idx);
            vocab_idx += 1;
        }

        // Add merged tokens (sorted by rank for deterministic order)
        let mut pairs: Vec<_> = bpe_ranks.iter().collect();
        pairs.sort_by_key(|&(_, rank)| rank);
        for (pair, _) in pairs {
            encoder.insert(format!("{}{}", pair.0, pair.1), vocab_idx);
            vocab_idx += 1;
        }

        // Add special tokens
        encoder.insert("<|startoftext|>".to_string(), START_OF_TEXT);
        encoder.insert("<|endoftext|>".to_string(), END_OF_TEXT);

        let decoder: HashMap<u32, String> = encoder.iter().map(|(k, &v)| (v, k.clone())).collect();

        let pat = Regex::new(
            r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
        )?;

        Ok(Self {
            byte_encoder,
            byte_decoder,
            encoder,
            decoder,
            bpe_ranks,
            cache: std::cell::RefCell::new(HashMap::new()),
            pat,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        // Normalize text
        let text = text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();

        // Find all matches
        for mat in self.pat.find_iter(&text) {
            let token = mat.as_str();

            // Convert to byte encoding
            let byte_encoded: String = token
                .bytes()
                .map(|b| self.byte_encoder.get(&b).copied().unwrap_or('?'))
                .collect();

            // Apply BPE
            let bpe_result = self.bpe(&byte_encoded);

            // Look up token IDs
            for bpe_token in bpe_result.split_whitespace() {
                if let Some(&id) = self.encoder.get(bpe_token) {
                    tokens.push(id);
                }
            }
        }

        tokens
    }

    /// Encode text with start/end tokens and padding to max_length
    pub fn encode_padded(&self, text: &str, max_length: usize) -> Vec<u32> {
        let mut tokens = vec![START_OF_TEXT];
        tokens.extend(self.encode(text));
        tokens.push(END_OF_TEXT);

        // Pad or truncate to max_length
        tokens.resize(max_length, 0);
        tokens
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let text: String = tokens
            .iter()
            .filter_map(|&id| self.decoder.get(&id))
            .cloned()
            .collect();

        // Convert byte encoding back to text
        let bytes: Vec<u8> = text
            .chars()
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect();

        String::from_utf8_lossy(&bytes)
            .replace("</w>", " ")
            .trim()
            .to_string()
    }

    /// Apply BPE to a token
    fn bpe(&self, token: &str) -> String {
        // Check cache
        if let Some(cached) = self.cache.borrow().get(token) {
            return cached.clone();
        }

        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        if word.is_empty() {
            return String::new();
        }

        // Add end-of-word marker to last character
        if let Some(last) = word.last_mut() {
            *last = format!("{}</w>", last);
        }

        // Iteratively merge pairs
        loop {
            // Find the pair with the lowest rank
            let pairs = get_pairs(&word);
            if pairs.is_empty() {
                break;
            }

            let min_pair = pairs
                .iter()
                .filter_map(|pair| self.bpe_ranks.get(pair).map(|&rank| (pair, rank)))
                .min_by_key(|&(_, rank)| rank);

            let Some((bigram, _)) = min_pair else {
                break;
            };

            // Merge the pair
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word.len() {
                if i < word.len() - 1 && word[i] == bigram.0 && word[i + 1] == bigram.1 {
                    new_word.push(format!("{}{}", bigram.0, bigram.1));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            word = new_word;
        }

        let result = word.join(" ");
        self.cache
            .borrow_mut()
            .insert(token.to_string(), result.clone());
        result
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }
}

impl Default for ClipTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse vocab.json content (simple JSON: {"token": id, ...})
fn parse_vocab_json(json: &str) -> Result<HashMap<String, u32>, TokenizerError> {
    let mut encoder = HashMap::new();

    // Simple JSON parser for {"key": num, "key2": num2, ...} format
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err(TokenizerError::InvalidVocab);
    }

    let content = &json[1..json.len() - 1];

    // State machine to parse key-value pairs
    let mut chars = content.chars().peekable();

    loop {
        // Skip whitespace and commas
        while chars.peek().is_some_and(|&c| c.is_whitespace() || c == ',') {
            chars.next();
        }

        if chars.peek().is_none() {
            break;
        }

        // Expect opening quote for key
        if chars.next() != Some('"') {
            return Err(TokenizerError::InvalidVocab);
        }

        // Read key (handle escape sequences)
        let mut key = String::new();
        loop {
            match chars.next() {
                Some('\\') => {
                    // Handle escape sequences
                    match chars.next() {
                        Some('n') => key.push('\n'),
                        Some('r') => key.push('\r'),
                        Some('t') => key.push('\t'),
                        Some('"') => key.push('"'),
                        Some('\\') => key.push('\\'),
                        Some(c) => {
                            key.push('\\');
                            key.push(c);
                        }
                        None => return Err(TokenizerError::InvalidVocab),
                    }
                }
                Some('"') => break,
                Some(c) => key.push(c),
                None => return Err(TokenizerError::InvalidVocab),
            }
        }

        // Skip whitespace and colon
        while chars.peek().is_some_and(|&c| c.is_whitespace()) {
            chars.next();
        }
        if chars.next() != Some(':') {
            return Err(TokenizerError::InvalidVocab);
        }
        while chars.peek().is_some_and(|&c| c.is_whitespace()) {
            chars.next();
        }

        // Read number
        let mut num_str = String::new();
        while chars.peek().is_some_and(|&c| c.is_ascii_digit()) {
            num_str.push(chars.next().unwrap());
        }

        let id: u32 = num_str.parse().map_err(|_| TokenizerError::InvalidVocab)?;
        encoder.insert(key, id);
    }

    Ok(encoder)
}

/// Parse BPE merges file
fn parse_bpe_merges(merges: &str) -> Result<HashMap<(String, String), usize>, TokenizerError> {
    let lines: Vec<&str> = merges.lines().collect();

    // Skip header line if present (starts with #version)
    let merges_start = if lines.first().is_some_and(|l| l.starts_with("#version")) {
        1
    } else {
        0
    };

    let bpe_ranks: HashMap<(String, String), usize> = lines[merges_start..]
        .iter()
        .enumerate()
        .filter_map(|(i, line)| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                Some(((parts[0].to_string(), parts[1].to_string()), i))
            } else {
                None
            }
        })
        .collect();

    Ok(bpe_ranks)
}

/// Get all adjacent pairs in a word
fn get_pairs(word: &[String]) -> Vec<(String, String)> {
    word.windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect()
}

/// Build byte-to-unicode mapping
///
/// CLIP uses a specific mapping where printable ASCII maps to itself,
/// and other bytes map to Unicode private use area.
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new();

    // Printable ASCII ranges
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xa1'..=b'\xac');
    bs.extend(b'\xae'..=b'\xff');

    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();

    // Map remaining bytes to private use area
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }

    bs.into_iter().zip(cs).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let mapping = bytes_to_unicode();
        assert_eq!(mapping.len(), 256);

        // Check printable ASCII maps to itself
        assert_eq!(mapping.get(&b'a'), Some(&'a'));
        assert_eq!(mapping.get(&b'Z'), Some(&'Z'));
        assert_eq!(mapping.get(&b'5'), Some(&'5'));
    }

    #[test]
    fn test_special_tokens() {
        assert_eq!(START_OF_TEXT, 49406);
        assert_eq!(END_OF_TEXT, 49407);
    }

    #[test]
    fn test_tokenizer_deterministic() {
        let tokenizer = ClipTokenizer::new();
        let tokens1 = tokenizer.encode_padded("a cute cat", 77);
        let tokens2 = tokenizer.encode_padded("a cute cat", 77);
        assert_eq!(tokens1, tokens2);
    }

    #[test]
    fn test_known_tokens() {
        let tokenizer = ClipTokenizer::new();
        let tokens = tokenizer.encode_padded("a cute cat", 77);
        // Verify against official CLIP vocab.json
        assert_eq!(tokens[0], START_OF_TEXT); // 49406
        assert_eq!(tokens[1], 320); // "a</w>"
        assert_eq!(tokens[2], 2242); // "cute</w>"
        assert_eq!(tokens[3], 2368); // "cat</w>"
        assert_eq!(tokens[4], END_OF_TEXT); // 49407
    }

    #[test]
    fn test_vocab_json_parsing() {
        let json = r#"{"hello": 1, "world": 2, "test</w>": 3}"#;
        let vocab = parse_vocab_json(json).unwrap();
        assert_eq!(vocab.get("hello"), Some(&1));
        assert_eq!(vocab.get("world"), Some(&2));
        assert_eq!(vocab.get("test</w>"), Some(&3));
    }
}
