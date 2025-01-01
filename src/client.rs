//! Client implementation for the Gemini AI API.

use crate::{
    error::GeminiError,
    models::{ModelParams, Request, RequestType, Response},
};

/// Default API endpoint for Google's Generative AI service
const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// Default API version
const DEFAULT_API_VERSION: &str = "v1beta";

/// A client for interacting with the Gemini AI API.
#[derive(Debug, Clone)]
pub struct GenerativeModel {
    api_key: String,
    params: ModelParams,
    client: reqwest::Client,
}

impl GenerativeModel {
    /// Creates a new GenerativeModel with the specified API key and model.
    ///
    /// # Arguments
    ///
    /// * `api_key` - The API key for authentication
    /// * `params` - The model parameters
    pub fn new(api_key: impl Into<String>, params: impl Into<ModelParams>) -> Self {
        Self {
            api_key: api_key.into(),
            params: params.into(),
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
    pub fn from_env(params: impl Into<ModelParams>) -> Result<Self, GeminiError> {
        let api_key = std::env::var("GOOGLE_API_KEY")?;
        Ok(Self::new(api_key, params))
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
    async fn make_request(&self, url: &str, request: Request) -> Result<Response, GeminiError> {
        let response = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await?
            .json::<Response>()
            .await?;

        Ok(response)
    }

    fn build_url(&self, request_type: RequestType) -> String {
        format!(
            "{}/{}/models/{}:{}?key={}",
            DEFAULT_BASE_URL, DEFAULT_API_VERSION, self.params.model, request_type, self.api_key
        )
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
    pub async fn send_message(&self, prompt: impl Into<String>) -> Result<Response, GeminiError> {
        let request = Request::with_prompt(prompt);
        let url = self.build_url(RequestType::GenerateContent);
        self.make_request(&url, request).await
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
    pub async fn generate_content(
        &self,
        request: impl Into<Request>,
    ) -> Result<Response, GeminiError> {
        let url = self.build_url(RequestType::GenerateContent);
        self.make_request(&url, request.into()).await
    }
}
