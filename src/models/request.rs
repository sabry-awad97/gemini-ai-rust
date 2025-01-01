//! Request models for the Gemini AI API.

use serde::Serialize;

use super::Part;

/// A request to the Gemini AI API.
#[derive(Debug, Clone, Serialize)]
pub struct Request {
    /// Optional system instruction for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<Content>,
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
            system_instruction: None,
            contents: vec![Content {
                parts: vec![Part::Text { text: text.into() }],
            }],
        }
    }

    /// Creates a new request with a system instruction and text prompt.
    ///
    /// # Arguments
    ///
    /// * `system_instruction` - The system instruction for the model
    /// * `text` - The text prompt to generate content from
    pub fn with_system_instruction(
        system_instruction: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            system_instruction: Some(Content {
                parts: vec![Part::Text {
                    text: system_instruction.into(),
                }],
            }),
            contents: vec![Content {
                parts: vec![Part::Text { text: text.into() }],
            }],
        }
    }
}
