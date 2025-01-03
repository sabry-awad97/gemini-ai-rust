use serde::{Deserialize, Serialize};

use super::{Content, Part, Role};

/// A system instruction for the model
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemInstruction {
    /// A content instruction
    Content(Content),
}

impl From<&str> for SystemInstruction {
    fn from(prompt: &str) -> Self {
        SystemInstruction::Content(Content {
            role: Some(Role::System),
            parts: vec![Part::Text {
                text: prompt.into(),
            }],
        })
    }
}
