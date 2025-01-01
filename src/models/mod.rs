//! Data structures for the Gemini AI API requests and responses.

mod part;
mod request;
mod response;

pub use part::Part;
pub use request::{Content, Request};
pub use response::{Candidate, CandidateContent, Response, UsageMetadata};
