//! Error types for the Gemini AI client.

use thiserror::Error;

/// Errors that can occur when using the Gemini AI client.
#[derive(Debug, Error)]
pub enum GoogleGenerativeAIError {
    /// Base error for the Gemini AI client.
    #[error("[GoogleGenerativeAI Error]: {message}")]
    Base {
        /// Error message
        message: String,
    },

    /// Error occurred during an API request.
    #[error("API request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    /// Error occurred when accessing environment variables.
    #[error("Environment variable not found: {0}")]
    EnvError(#[from] std::env::VarError),

    /// Error occurred when parsing JSON.
    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl GoogleGenerativeAIError {
    /// Creates a new Base error with the given message.
    pub fn new(message: impl Into<String>) -> Self {
        Self::Base {
            message: message.into(),
        }
    }
}
