use std::fmt;

/// The type of request to make to the API.
#[derive(Debug, Copy, Clone)]
pub enum RequestType {
    /// A request to generate content.
    GenerateContent,
    /// A request to generate content in a streaming fashion.
    StreamGenerateContent,
}

impl fmt::Display for RequestType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GenerateContent => write!(f, "generateContent"),
            Self::StreamGenerateContent => write!(f, "streamGenerateContent"),
        }
    }
}
