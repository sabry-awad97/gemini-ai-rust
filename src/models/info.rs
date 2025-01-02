use serde::{Deserialize, Serialize};

/// Information about a Gemini model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    /// The name of the model.
    pub name: String,
    /// A description of what the model does.
    pub description: String,
    /// The display name of the model.
    pub display_name: String,
    /// Maximum number of tokens allowed for input.
    pub input_token_limit: i32,
    /// Maximum number of tokens allowed for output.
    pub output_token_limit: i32,
    /// List of supported generation methods (e.g., generateContent, countTokens).
    pub supported_generation_methods: Vec<String>,
    /// Default temperature for sampling from output distribution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum temperature allowed for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_temperature: Option<f32>,
    /// Default top_p for nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Default top_k for sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    /// Version of the model.
    pub version: String,
}
