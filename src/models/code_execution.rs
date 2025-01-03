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

/// Result of code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CodeExecutionResult {
    /// The outcome of the code execution.
    pub outcome: CodeExecutionOutcome,
    /// The output produced by the code execution.
    pub output: String,
}

/// Possible outcomes of code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CodeExecutionOutcome {
    /// Code executed successfully.
    #[serde(rename = "OUTCOME_OK")]
    Ok,
    /// Code execution failed.
    #[serde(rename = "OUTCOME_ERROR")]
    Error,
    /// Code execution was blocked.
    #[serde(rename = "OUTCOME_BLOCKED")]
    Blocked,
}

/// Represents executable code in a specific programming language.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutableCode {
    /// The programming language of the code.
    pub language: String,
    /// The actual code to be executed.
    pub code: String,
}
