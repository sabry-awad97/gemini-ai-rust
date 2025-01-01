//! Request models for the Gemini AI API.

use serde::Serialize;
use typed_builder::TypedBuilder;

use super::{model_params::GenerationConfig, Part, SafetySetting};

/// A request to the Gemini AI API.
#[derive(Debug, Clone, Serialize, TypedBuilder)]
#[builder(doc)]
pub struct Request {
    /// The contents of the request, including the prompt text.
    #[builder(setter(into))]
    pub contents: Vec<Content>,

    /// Optional configuration for text generation
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub generation_config: Option<GenerationConfig>,

    /// Optional system instruction for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub system_instruction: Option<Content>,

    /// Optional safety settings for content filtering
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub safety_settings: Option<Vec<SafetySetting>>,
}

/// A content object containing parts of the request.
#[derive(Debug, Clone, Serialize)]
pub struct Content {
    /// The parts that make up the content.
    pub parts: Vec<Part>,
}

impl Request {
    /// Creates a new request with the given text prompt.
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt to generate content from
    pub fn with_prompt(text: impl Into<String>) -> Self {
        Self::builder()
            .contents(vec![Content {
                parts: vec![Part::Text { text: text.into() }],
            }])
            .build()
    }
}
