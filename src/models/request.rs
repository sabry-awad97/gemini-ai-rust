//! Request models for the Gemini AI API.

use serde::Serialize;

use super::Part;

/// A request to the Gemini AI API.
#[derive(Debug, Clone, Serialize)]
pub struct Request {
    /// The contents of the request, including the prompt text.
    pub contents: Vec<Content>,
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
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            contents: vec![Content {
                parts: vec![Part { text: text.into() }],
            }],
        }
    }
}
