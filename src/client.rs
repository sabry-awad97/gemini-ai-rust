//! Client implementation for the Gemini AI API.

use crate::{
    error::GeminiError,
    models::{Request, Response, SafetySetting},
};

/// Default API endpoint for Google's Generative AI service
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// Default API version
const DEFAULT_API_VERSION: &str = "v1beta";

/// A client for interacting with the Gemini AI API.
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    api_key: String,
    model: String,
    client: reqwest::Client,
}

impl GenerativeModel {
    /// Creates a new GenerativeModel with the specified API key and model.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key for authentication
    /// * `model` - The model identifier (e.g., "gemini-1.5-flash")
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }

    /// Creates a new GenerativeModel from environment variables.
    ///
    /// # Environment Variables
    ///
    /// * `GOOGLE_API_KEY` - The API key for authentication
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "gemini-1.5-flash")
    ///
    /// # Errors
    ///
    /// Returns an error if the required environment variable is not set.
    pub fn from_env(model: impl Into<String>) -> Result<Self, GeminiError> {
        let api_key = std::env::var("GOOGLE_API_KEY")?;
        Ok(Self::new(api_key, model))
    }

    /// Makes a request to the Gemini AI API.
    ///
    /// # Arguments
    ///
    /// * `request` - The request to send to the API
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    async fn make_request(&self, request: Request) -> Result<Response, GeminiError> {
        let url = format!(
            "{}/{}/models/{}:generateContent?key={}",
            DEFAULT_BASE_URL, DEFAULT_API_VERSION, self.model, self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .json::<Response>()
            .await?;

        Ok(response)
    }

    /// Generates content using the Gemini AI API.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt to generate content from
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    pub async fn generate_content(
        &self,
        prompt: impl Into<String>,
    ) -> Result<Response, GeminiError> {
        let request = Request::new(prompt);
        self.make_request(request).await
    }

    /// Generates content using the Gemini AI API with a system instruction.
    ///
    /// # Arguments
    ///
    /// * `system_instruction` - The system instruction for the model
    /// * `prompt` - The text prompt to generate content from
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    pub async fn generate_content_with_system(
        &self,
        system_instruction: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<Response, GeminiError> {
        let request = Request::with_system_instruction(system_instruction, prompt);
        self.make_request(request).await
    }

    /// Generates content using the Gemini AI API with safety settings.
    ///
    /// # Arguments
    ///
    /// * `prompt` - The text prompt to generate content from
    /// * `safety_settings` - List of safety settings to apply
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    pub async fn generate_content_with_safety(
        &self,
        prompt: impl Into<String>,
        safety_settings: Vec<SafetySetting>,
    ) -> Result<Response, GeminiError> {
        let request = Request::with_safety_settings(prompt, safety_settings);
        self.make_request(request).await
    }

    /// Generates content using the Gemini AI API with both system instruction and safety settings.
    ///
    /// # Arguments
    ///
    /// * `system_instruction` - The system instruction for the model
    /// * `prompt` - The text prompt to generate content from
    /// * `safety_settings` - List of safety settings to apply
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails or if the response cannot be parsed.
    pub async fn generate_content_with_system_and_safety(
        &self,
        system_instruction: impl Into<String>,
        prompt: impl Into<String>,
        safety_settings: Vec<SafetySetting>,
    ) -> Result<Response, GeminiError> {
        let request = Request::with_system_and_safety(system_instruction, prompt, safety_settings);
        self.make_request(request).await
    }
}
