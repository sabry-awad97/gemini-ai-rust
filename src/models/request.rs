//! Request models for the Gemini AI API.

use serde::{Deserialize, Serialize};
use typed_builder::TypedBuilder;

use super::{
    model_params::GenerationConfig, system_instruction::SystemInstruction, tool::ToolConfig, Part,
    SafetySetting, Tool,
};

/// A request to the Gemini AI API.
#[derive(Debug, Clone, Serialize, TypedBuilder)]
#[builder(doc)]
pub struct Request {
    /// The contents of the request, including the prompt text.
    #[builder(setter(into))]
    pub contents: Vec<Content>,

    /// Optional configuration for text generation
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub generation_config: Option<GenerationConfig>,

    /// Optional system instruction for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(into))]
    pub system_instruction: Option<SystemInstruction>,

    /// Optional safety settings for content filtering
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub safety_settings: Option<Vec<SafetySetting>>,

    /// Optional tools for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub tools: Option<Vec<Tool>>,

    /// Optional configuration for function calling
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub tool_config: Option<ToolConfig>,

    /// Optional cached content for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option, into))]
    pub cached_content: Option<String>,
}

/// Role of a participant in a chat
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The user role
    User,
    /// The model role
    Model,
    /// The system role
    System,
    /// The function role for function responses
    Function,
}

/// A content object containing parts of the request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// The role of the content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    /// The parts that make up the content.
    pub parts: Vec<Part>,
}

impl Request {
    /// Creates a new request with the given text prompt.
    ///
    /// # Arguments
    ///
    /// * `text` - The text prompt to generate content from
    pub fn with_prompt(text: impl Into<String>) -> Self {
        Self::builder()
            .contents(vec![Content {
                role: Some(Role::User),
                parts: vec![Part::Text { text: text.into() }],
            }])
            .build()
    }
}

/// Request structure for the embedContent API endpoint
#[derive(Debug, Serialize)]
pub struct EmbedContentRequest {
    /// The content to generate embeddings for
    pub content: Content,
    /// Optional task type to optimize the embedding for
    pub task_type: Option<TaskType>,
}

impl EmbedContentRequest {
    /// Creates a new EmbedContentRequest with the given prompt and optional task type.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt to generate embeddings for
    /// * `task_type` - The task type to optimize the embedding for
    ///
    /// # Returns
    ///
    /// A new EmbedContentRequest
    pub fn new(prompt: &str, task_type: Option<TaskType>) -> Self {
        Self {
            content: Content {
                role: None,
                parts: vec![Part::Text {
                    text: prompt.to_string(),
                }],
            },
            task_type,
        }
    }
}

/// Type of task for which the embedding will be used
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum TaskType {
    /// Optimize embedding for retrieval queries
    RetrievalQuery,
    /// Optimize embedding for retrieval documents
    RetrievalDocument,
    /// Optimize embedding for semantic similarity tasks
    SemanticSimilarity,
    /// Optimize embedding for classification tasks
    Classification,
    /// Optimize embedding for clustering tasks
    Clustering,
}
