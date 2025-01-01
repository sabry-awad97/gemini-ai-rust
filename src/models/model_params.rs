use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

/// Configuration parameters for the generative model
#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[builder(doc)]
pub struct ModelParams {
    /// Model identifier (e.g., "gemini-1.5-flash")
    #[builder(setter(into), default = String::from("gemini-1.5-flash"))]
    pub model: String,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self::builder().build()
    }
}
