use serde::{Deserialize, Serialize};

/// A tool that enables the model to execute code as part of generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionTool {
    /// Empty object to enable code execution. This field may have subfields added in the future.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code_execution: Option<CodeExecutionConfig>,
}

/// Configuration for code execution.
/// Currently an empty struct as per API specification, but may have fields added in the future.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionConfig {}
