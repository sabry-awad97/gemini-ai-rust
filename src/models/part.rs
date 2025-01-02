//! Common part model used in both requests and responses.

use base64::{engine::general_purpose::STANDARD as base64_engine, Engine};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A part containing text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    /// A text part containing a string value
    Text {
        /// The text content of the part
        text: String,
    },
    /// A part containing inline data
    InlineData {
        /// The inline data content of the part
        inline_data: InlineData,
    },

    /// A part containing file data
    FileData {
        /// The file data content of the part
        file_data: FileData,
    },

    /// A part containing a function call
    FunctionCall {
        /// The function call content of the part
        function_call: FunctionCall,
    },
    /// A part containing a function response
    FunctionResponse {
        /// The function response content of the part
        function_response: FunctionResponse,
    },
}

impl Part {
    /// Creates a new text part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Creates a new inline data part from a file path.
    pub fn image_from_path(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        let mime_type = mime_guess::from_path(path)
            .first_or_octet_stream()
            .to_string();
        let data = base64_engine.encode(data);
        Ok(Self::InlineData {
            inline_data: InlineData { mime_type, data },
        })
    }

    /// Creates a new file data part.
    pub fn file_data(mime_type: impl Into<String>, file_uri: impl Into<String>) -> Self {
        Self::FileData {
            file_data: FileData {
                mime_type: mime_type.into(),
                file_uri: file_uri.into(),
            },
        }
    }
}

/// Inline data (base64 encoded)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineData {
    /// The MIME type of the inline data
    pub mime_type: String,
    /// Base64 encoded data
    pub data: String,
}

/// File data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    /// The MIME type of the file data
    pub mime_type: String,
    /// The URI of the file
    pub file_uri: String,
}

/// Function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The name of the function
    pub name: String,
    /// The arguments of the function
    pub args: serde_json::Value,
}

/// Function response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// The name of the function
    pub name: String,
    /// The response of the function
    pub response: serde_json::Value,
}
