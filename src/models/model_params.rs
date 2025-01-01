use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::ResponseSchema;

/// Parameters for configuring text generation
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(doc)]
pub struct GenerationConfig {
    /// Number of candidate responses to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub candidate_count: Option<i32>,

    /// Sequences to stop generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub stop_sequences: Option<Vec<String>>,

    /// Maximum number of tokens to generate in the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub max_output_tokens: Option<i32>,

    /// Sampling temperature to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub temperature: Option<f32>,

    /// Top-p sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub top_p: Option<f32>,

    /// Top-k sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub top_k: Option<i32>,

    /// Output response MIME type of the generated candidate text.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub response_mime_type: Option<String>,

    /// Output response schema of the generated candidate text.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub response_schema: Option<ResponseSchema>,

    /// Presence penalty applied to the next token's logprobs if the token has already been seen in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub presence_penalty: Option<f32>,

    /// Frequency penalty applied to the next token's logprobs, multiplied by the number of times each token has been seen in the response so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub frequency_penalty: Option<f32>,

    /// If true, export the logprobs results in response.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub response_logprobs: Option<bool>,

    /// Valid if response_logprobs is set to true. This will set the number of top logprobs to return at each decoding step in the logprobsResult.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub logprobs: Option<i32>,
}

/// Parameters for configuring a generative model.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(doc)]
pub struct ModelParams {
    /// The model identifier (e.g., "gemini-1.5-flash")
    #[builder(default = "gemini-1.5-flash".to_string(), setter(into))]
    pub model: String,

    /// Optional configuration for text generation
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub generation_config: Option<GenerationConfig>,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self::builder().build()
    }
}
