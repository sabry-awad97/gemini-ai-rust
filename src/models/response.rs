//! Response models for the Gemini AI API.

use serde::{Deserialize, Serialize};

use super::{Content, FunctionCall, HarmCategory, ModelInfo, Part};

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

    /// Returns a vector of function calls from all candidates in the response.
    ///
    /// This method collects all function calls from the response candidates and returns them
    /// as a vector. If there are no function calls in the response, an empty vector is returned.
    pub fn function_calls(&self) -> Vec<FunctionCall> {
        self.candidates
            .iter()
            .flat_map(|candidate| {
                candidate
                    .content
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        Part::FunctionCall { function_call } => Some(function_call.clone()),
                        _ => None,
                    })
            })
            .collect()
    }

    /// Gets all executable code parts from the response.
    pub fn executable_code(&self) -> Vec<ExecutableCode> {
        self.candidates
            .iter()
            .flat_map(|candidate| {
                candidate
                    .content
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        Part::ExecutableCode { executable_code } => Some(executable_code.clone()),
                        _ => None,
                    })
            })
            .collect()
    }

    /// Gets all code execution results from the response.
    pub fn code_execution_results(&self) -> Vec<CodeExecutionResult> {
        self.candidates
            .iter()
            .flat_map(|candidate| {
                candidate
                    .content
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        Part::CodeExecutionResult {
                            code_execution_result,
                        } => Some(code_execution_result.clone()),
                        _ => None,
                    })
            })
            .collect()
    }
}

/// A candidate response from the model.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// The content of the candidate response.
    pub content: Content,
    /// The reason why the generation finished.
    pub finish_reason: Option<FinishReason>,
    /// Safety ratings for different harm categories.
    pub safety_ratings: Option<Vec<SafetyRating>>,

    /// Citation information for this candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_metadata: Option<CitationMetadata>,

    /// Average log probabilities for the generation.
    pub avg_logprobs: Option<f64>,
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

/// Citation metadata for a candidate.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CitationMetadata {
    /// The citations for this candidate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<Vec<Citation>>,
}

/// A citation for a candidate.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Citation {
    /// The start index of the citation.
    pub start_index: i32,

    /// The end index of the citation.
    pub end_index: i32,

    /// The URI of the citation.
    pub uri: String,

    /// The license of the citation.
    pub license: String,
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

/// Reason why the generation finished.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    #[serde(rename = "FINISH_REASON_UNSPECIFIED")]
    /// Default value. This value is unused.
    Unspecified,
    /// Natural stop point of the model or provided stop sequence.
    Stop,
    /// The maximum number of tokens as specified in the request was reached.
    MaxTokens,
    /// The response candidate content was flagged for safety reasons.
    Safety,
    /// The response candidate content was flagged for recitation reasons.
    Recitation,
    /// The response candidate content was flagged for using an unsupported language.
    Language,
    /// Unknown reason.
    Other,
    /// Token generation stopped because the content contains forbidden terms.
    Blocklist,
    /// Token generation stopped for potentially containing prohibited content.
    ProhibitedContent,
    /// Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).
    Spii,
    /// The function call generated by the model is invalid.
    MalformedFunctionCall,
}

/// Metadata about token usage in the request and response.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Number of tokens in the prompt.
    pub prompt_token_count: i32,
    /// Number of tokens in the generated candidates.
    pub candidates_token_count: Option<i32>,
    /// Total number of tokens used.
    pub total_token_count: i32,
}

/// Response from token counting.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenCountResponse {
    /// Total number of tokens in the request.
    pub total_tokens: i32,
}

/// Response from listing available models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// List of available models and their details.
    pub models: Vec<ModelInfo>,
    /// Token for retrieving the next page of results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_page_token: Option<String>,
}

/// Represents executable code in a specific programming language.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutableCode {
    /// The programming language of the code.
    pub language: String,
    /// The actual code to be executed.
    pub code: String,
}

/// Result of code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecutionResult {
    /// The outcome of the code execution.
    pub outcome: CodeExecutionOutcome,
    /// The output produced by the code execution.
    pub output: String,
}

/// Possible outcomes of code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CodeExecutionOutcome {
    /// Code executed successfully.
    #[serde(rename = "OUTCOME_OK")]
    Ok,
    /// Code execution failed.
    #[serde(rename = "OUTCOME_ERROR")]
    Error,
    /// Code execution was blocked.
    #[serde(rename = "OUTCOME_BLOCKED")]
    Blocked,
}
