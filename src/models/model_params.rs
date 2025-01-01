use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
pub struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default = Some(0.7), setter(into))]
    pub temperature: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default = Some(40), setter(into))]
    pub top_k: Option<i32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default = Some(0.95), setter(into))]
    pub top_p: Option<f32>,
}

/// Parameters for configuring a generative model.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(doc)]
pub struct ModelParams {
    /// The model identifier (e.g., "gemini-1.5-flash")
    #[builder(default = "gemini-1.5-flash".to_string(), setter(into))]
    pub model: String,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self::builder().build()
    }
}
