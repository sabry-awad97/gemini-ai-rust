//! Data structures for the Gemini AI API requests and responses.

mod model_params;
mod part;
mod request;
mod request_type;
mod response;
mod safety;
mod schema;
mod stream;

pub use model_params::{GenerationConfig, ModelParams};
pub use part::Part;
pub use request::{Content, Request, Role};
pub use request_type::RequestType;
pub use response::{Candidate, Response, SafetyProbability, SafetyRating, UsageMetadata};
pub use safety::{HarmCategory, SafetySetting, SafetyThreshold};
pub use schema::SchemaType;
pub use stream::ResponseStream;

/// Alias for the Schema type
pub type ResponseSchema = schema::Schema;
