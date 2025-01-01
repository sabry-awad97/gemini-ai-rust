//! Response models for the Gemini AI API.

use serde::Deserialize;

use super::Part;

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

/// A candidate response from the model.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// The content of the candidate response.
    pub content: CandidateContent,
    /// The reason why the generation finished.
    pub finish_reason: String,
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

impl Response {
    /// Displays the response in a formatted way.
    pub fn display(&self) {
        println!("Response:");
        println!("Model Version: {}", self.model_version);
        println!("\nGenerated Text:");
        for candidate in &self.candidates {
            println!("Role: {}", candidate.content.role);
            for part in &candidate.content.parts {
                println!("{}", part.text);
            }
            println!("\nFinish Reason: {}", candidate.finish_reason);
            println!("Average Log Probability: {}", candidate.avg_logprobs);
        }

        println!("\nUsage Statistics:");
        println!("Prompt Tokens: {}", self.usage_metadata.prompt_token_count);
        println!(
            "Response Tokens: {}",
            self.usage_metadata.candidates_token_count
        );
        println!("Total Tokens: {}", self.usage_metadata.total_token_count);
    }
}
