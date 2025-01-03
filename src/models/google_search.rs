use serde::{Deserialize, Serialize};

/// Tool that enables Google search retrieval.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSearchTool {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Google search retrieval tool config.
    pub google_search: Option<GoogleSearch>,
}

/// Configuration for Google search retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoogleSearch {}
