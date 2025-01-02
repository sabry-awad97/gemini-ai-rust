use serde::{Deserialize, Serialize};

use super::function::FunctionDeclaration;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
/// Tools data sent as part of the request.
pub enum Tool {
    /// A list of function declarations.
    ///
    /// This can be used to define functions that can be used in the chat.
    FunctionDeclarationsTool {
        /// The list of function declarations
        function_declarations: Vec<FunctionDeclaration>,
    },
}

impl Tool {
    /// Creates a new function declarations tool.
    ///
    /// # Arguments
    ///
    /// * `function_declarations` - The list of function declarations
    ///   to include in the tool
    pub fn function_declarations(function_declarations: Vec<FunctionDeclaration>) -> Self {
        Self::FunctionDeclarationsTool {
            function_declarations,
        }
    }
}

impl From<Vec<FunctionDeclaration>> for Tool {
    fn from(function_declarations: Vec<FunctionDeclaration>) -> Self {
        Self::FunctionDeclarationsTool {
            function_declarations,
        }
    }
}
