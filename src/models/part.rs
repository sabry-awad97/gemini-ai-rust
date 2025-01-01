//! Common part model used in both requests and responses.

use serde::{Deserialize, Serialize};

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
}

/// A part containing inline data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InlineData {
    /// The MIME type of the inline data
    pub mime_type: String,
    /// The inline data content
    pub data: String,
}
