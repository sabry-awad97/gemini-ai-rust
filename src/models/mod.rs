//! Data structures for the Gemini AI API requests and responses.

mod code_execution;
mod function;
mod google_search;
mod grounding_metadata;
mod info;
mod model_params;
mod part;
mod request;
mod request_type;
mod response;
mod safety;
mod schema;
mod stream;
mod system_instruction;
mod tool;

pub use code_execution::{CodeExecutionConfig, CodeExecutionResult, CodeExecutionOutcome, CodeExecutionTool};
pub use function::{
    FunctionCall, FunctionCallingConfig, FunctionCallingMode, FunctionDeclaration,
    FunctionDeclarationSchema, FunctionResponse,
};
pub use google_search::GoogleSearch;
pub use info::ModelInfo;
pub use model_params::{GenerationConfig, ModelParams};
pub use part::Part;
pub use request::{Content, Request, Role};
pub use request_type::RequestType;
pub use response::{
    Candidate, ListModelsResponse, Response, SafetyProbability, SafetyRating, TokenCountResponse,
    UsageMetadata,
};
pub use safety::{HarmCategory, SafetySetting, SafetyThreshold};
pub use schema::{Schema, SchemaType};
pub use stream::ResponseStream;
pub use system_instruction::SystemInstruction;
pub use tool::{Tool, ToolConfig};

/// Alias for the Schema type
pub type ResponseSchema = schema::Schema;
