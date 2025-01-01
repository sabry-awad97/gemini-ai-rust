use std::fmt::Display;

/// Type of request to be made to the Gemini AI API.
#[derive(Debug, Clone, Copy)]
pub enum RequestType {
    /// Generate content from a prompt
    GenerateContent,
    /// Stream content from a prompt
    StreamGenerateContent,
    /// Count tokens in a prompt
    CountTokens,
    /// Embed content
    EmbedContent,
}

impl Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GenerateContent => write!(f, "generateContent"),
            Self::StreamGenerateContent => write!(f, "streamGenerateContent"),
            Self::CountTokens => write!(f, "countTokens"),
            Self::EmbedContent => write!(f, "embedContent"),
        }
    }
}
