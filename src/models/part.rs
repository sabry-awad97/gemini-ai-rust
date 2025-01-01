//! Common part model used in both requests and responses.

use serde::{Deserialize, Serialize};

/// A part containing text content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Part {
    /// The text content of the part.
    pub text: String,
}
