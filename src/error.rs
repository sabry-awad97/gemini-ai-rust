//! Error types for the Gemini AI client.

use thiserror::Error;

/// Errors that can occur when using the Gemini AI client.
#[derive(Debug, Error)]
pub enum GeminiError {
    /// Error occurred during an API request.
    #[error("API request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    /// Error occurred when accessing environment variables.
    #[error("Environment variable not found: {0}")]
    EnvError(#[from] std::env::VarError),
}
