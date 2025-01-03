use serde::{Deserialize, Serialize};

/// Metadata returned to client when grounding is enabled.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingMetadata {
    /// Google search entry for the following-up web searches.
    pub search_entry_point: Option<SearchEntryPoint>,
    /// List of supporting references retrieved from specified grounding source.
    pub grounding_chunks: Option<Vec<GroundingChunk>>,
    /// List of grounding support.
    pub grounding_supports: Option<Vec<GroundingSupport>>,
    /// Metadata related to retrieval in the grounding flow.
    pub retrieval_metadata: Option<RetrievalMetadata>,
    /// Web search queries for the following-up web search.
    pub web_search_queries: Option<Vec<String>>,
}

/// Google search entry point.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchEntryPoint {
    /// Web content snippet that can be embedded in a web page or an app webview.
    pub rendered_content: Option<String>,
    /// Base64 encoded JSON representing array of <search term, search url> tuple.
    pub sdk_blob: Option<String>,
}

/// Grounding chunk.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingChunk {
    /// Chunk from the web.
    pub web: Option<GroundingChunkWeb>,
}

/// Web chunk.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingChunkWeb {
    /// URI of the web page.
    pub uri: Option<String>,
    /// Title of the web page.
    pub title: Option<String>,
}

/// Grounding support
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupport {
    /// The segment of the content.
    pub segment: Option<GroundingSupportSegment>,
    /// A list of indices (into 'grounding_chunk') specifying the citations
    /// associated with the claim. For instance [1, 3, 4] means that
    /// grounding_chunk[1], grounding_chunk[3], grounding_chunk[4] are the
    /// retrieved content attributed to the claim.
    pub grounding_chunk_indices: Option<Vec<i64>>,
    /// Confidence score of the support references. Ranges from 0 to 1. 1 is the
    /// most confident. This list must have the same size as the
    /// grounding_chunk_indices.
    pub confidence_scores: Option<Vec<f64>>,
}

/// Metadata related to retrieval in the grounding flow
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalMetadata {
    /// Score indicating how likely information from google search could help
    /// answer the prompt. The score is in the range [0, 1], where 0 is the least
    /// likely and 1 is the most likely. This score is only populated when google
    /// search grounding and dynamic retrieval is enabled. It will be compared to
    /// the threshold to determine whether to trigger google search.
    pub google_search_dynamic_retrieval_score: Option<f64>,
}

/// Segment of the content
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingSupportSegment {
    /// The index of a Part object within its parent Content object.
    pub part_index: Option<i64>,
    /// Start index in the given Part, measured in bytes. Offset from the start of
    /// the Part, inclusive, starting at zero.
    pub start_index: Option<i64>,
    /// End index in the given Part, measured in bytes. Offset from the start of
    /// the Part, exclusive, starting at zero.
    pub end_index: Option<i64>,
    /// The text corresponding to the segment from the response.
    pub text: Option<String>,
}
