use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::{
    code_execution::{CodeExecutionConfig, CodeExecutionTool},
    function::{FunctionCallingConfig, FunctionDeclaration, FunctionDeclarationTool},
};

/// Configuration for tool behavior in the model.
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase")]
pub struct ToolConfig {
    /// Configuration for function calling behavior.
    #[builder(setter(into))]
    pub function_calling_config: FunctionCallingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
/// Tools data sent as part of the request.
pub enum Tool {
    /// A list of function declarations.
    ///
    /// This can be used to define functions that can be used in the chat.
    FunctionDeclarationsTool(FunctionDeclarationTool),
    /// Tool that enables code execution.
    CodeExecutionTool(CodeExecutionTool),
}

impl Tool {
    /// Creates a new function declarations tool.
    ///
    /// # Arguments
    ///
    /// * `function_declarations` - The list of function declarations
    ///   to include in the tool
    pub fn function_declarations(function_declarations: Vec<FunctionDeclaration>) -> Self {
        Self::FunctionDeclarationsTool(FunctionDeclarationTool {
            function_declarations,
        })
    }

    /// Default code execution tool with empty configuration.
    pub const CODE_EXECUTION: Self = Self::CodeExecutionTool(CodeExecutionTool {
        code_execution: Some(CodeExecutionConfig {}),
    });
}

impl From<Vec<FunctionDeclaration>> for Tool {
    fn from(function_declarations: Vec<FunctionDeclaration>) -> Self {
        Self::FunctionDeclarationsTool(FunctionDeclarationTool {
            function_declarations,
        })
    }
}

impl From<CodeExecutionTool> for Tool {
    fn from(tool: CodeExecutionTool) -> Self {
        Self::CodeExecutionTool(tool)
    }
}
