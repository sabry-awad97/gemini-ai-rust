use std::fmt::Display;

/// Type of request to be made to the API.
#[derive(Debug, Clone, Copy)]
pub enum RequestType {
    /// Generate content
    GenerateContent,
    /// Stream generate content
    StreamGenerateContent,
    /// Count tokens in content
    CountTokens,
    /// Embed content
    EmbedContent,
    /// Batch embed contents
    BatchEmbedContents,
}

impl Display for RequestType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GenerateContent => write!(f, "generateContent"),
            Self::StreamGenerateContent => write!(f, "streamGenerateContent"),
            Self::CountTokens => write!(f, "countTokens"),
            Self::EmbedContent => write!(f, "embedContent"),
            Self::BatchEmbedContents => write!(f, "batchEmbedContents"),
        }
    }
}
