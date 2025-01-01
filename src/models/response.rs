//! Response models for the Gemini AI API.

use serde::Deserialize;

use super::{HarmCategory, Part};

/// A response from the Gemini AI API.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Response {
    /// The generated candidates from the model.
    pub candidates: Vec<Candidate>,
    /// Metadata about token usage.
    pub usage_metadata: UsageMetadata,
    /// The version of the model used.
    pub model_version: String,
}

impl Response {
    /// Gets the text content from the first candidate's first part.
    pub fn text(&self) -> String {
        self.candidates
            .iter()
            .flat_map(|candidate| {
                candidate
                    .content
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        Part::Text { text } => Some(text.clone()),
                        _ => None,
                    })
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// A candidate response from the model.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// The content of the candidate response.
    pub content: CandidateContent,
    /// The reason why the generation finished.
    pub finish_reason: String,
    /// Safety ratings for different harm categories.
    pub safety_ratings: Vec<SafetyRating>,
    /// Average log probabilities for the generation.
    pub avg_logprobs: f64,
}

/// The content of a candidate response.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateContent {
    /// The parts that make up the content.
    pub parts: Vec<Part>,
    /// The role of the content generator (e.g., "model").
    pub role: String,
}

/// Safety rating for a specific harm category.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    /// The category of harm being rated.
    pub category: HarmCategory,
    /// The probability level of harmful content.
    pub probability: SafetyProbability,
}

/// Probability level for safety ratings.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyProbability {
    /// Negligible probability of harmful content.
    Negligible,
    /// Low probability of harmful content.
    Low,
    /// Medium probability of harmful content.
    Medium,
    /// High probability of harmful content.
    High,
}

/// Metadata about token usage in the request and response.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Number of tokens in the prompt.
    pub prompt_token_count: i32,
    /// Number of tokens in the generated candidates.
    pub candidates_token_count: i32,
    /// Total number of tokens used.
    pub total_token_count: i32,
}
