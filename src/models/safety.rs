use serde::{Deserialize, Serialize};

/// Safety category for content filtering in the Gemini AI API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    /// Content that harasses, bullies, or threatens individuals or groups.
    HarmCategoryHarassment,
    /// Content that expresses hateful, biased, or discriminatory views.
    HarmCategoryHateSpeech,
    /// Content of a sexual nature or containing explicit material.
    HarmCategorySexuallyExplicit,
    /// Content that promotes or provides instructions for dangerous activities.
    HarmCategoryDangerousContent,
    /// Content that may undermine or manipulate civic processes and institutions.
    HarmCategoryCivicIntegrity,
}

/// Safety threshold level for content filtering
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SafetyThreshold {
    /// Block only high-risk content
    BlockOnlyHigh,
    /// Block medium and high-risk content
    BlockMediumAndAbove,
    /// Block all potentially risky content
    BlockLowAndAbove,
    /// Block content with any risk level
    BlockNone,
    /// Allow all content regardless of risk
    UnspecifiedBlockThreshold,
}

/// Safety setting for a specific harm category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    /// The category of harm to filter
    pub category: HarmCategory,
    /// The threshold level for filtering
    pub threshold: SafetyThreshold,
}
