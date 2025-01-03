//! Chat session management for the Gemini AI API.

use crate::{
    error::GoogleGenerativeAIError,
    models::{Content, Part, Request, ResponseStream, Role, SystemInstruction},
    GenerativeModel,
};

/// A chat session with the Gemini AI model.
#[derive(Debug)]
pub struct ChatSession {
    /// The model client
    model: GenerativeModel,
    /// Chat history
    history: Vec<Content>,
    /// System instruction for the chat
    system_instruction: Option<SystemInstruction>,
}

impl ChatSession {
    /// Creates a new chat session.
    ///
    /// # Arguments
    ///
    /// * `model` - The Gemini AI model to use
    pub fn new(model: GenerativeModel) -> Self {
        Self {
            model,
            history: Vec::new(),
            system_instruction: None,
        }
    }

    /// Sets a system instruction for the chat session.
    ///
    /// # Arguments
    ///
    /// * `instruction` - The system instruction text
    pub fn with_system_instruction(mut self, instruction: impl Into<String>) -> Self {
        self.system_instruction = Some(SystemInstruction::Content(Content {
            role: Some(Role::System),
            parts: vec![Part::text(instruction.into())],
        }));
        self
    }

    /// Sends a message to the chat and gets a response.
    ///
    /// # Arguments
    ///
    /// * `message` - The message text to send
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    pub async fn send_message(
        &mut self,
        message: impl Into<String>,
    ) -> Result<String, GoogleGenerativeAIError> {
        let user_message = Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: message.into(),
            }],
        };

        // Build the complete message history
        let mut messages = Vec::new();
        messages.extend(self.history.clone());
        messages.push(user_message.clone());

        // Create the request
        let request = Request::builder()
            .system_instruction(self.system_instruction.as_ref().cloned())
            .contents(messages)
            .build();

        // Send the request
        let response = self.model.generate_response(request).await?;

        // Extract the response text
        if let Some(candidate) = response.candidates.first() {
            if let Some(Part::Text { text }) = candidate.content.parts.first() {
                // Update history
                self.history.push(user_message);
                self.history.push(candidate.content.clone());
                return Ok(text.clone());
            }
        }

        Err(GoogleGenerativeAIError::new(
            "No valid response from the model".to_string(),
        ))
    }

    /// Starts a streaming chat session.
    ///
    /// # Arguments
    ///
    /// * `message` - The message text to send
    ///
    /// # Errors
    ///
    /// Returns an error if the API request fails.
    pub async fn send_message_streaming(
        &mut self,
        message: impl Into<String>,
    ) -> Result<ResponseStream, GoogleGenerativeAIError> {
        let user_message = Content {
            role: Some(Role::User),
            parts: vec![Part::Text {
                text: message.into(),
            }],
        };

        // Build the complete message history
        let mut messages = Vec::new();
        messages.extend(self.history.clone());
        messages.push(user_message.clone());

        // Create the request
        let request = Request::builder()
            .system_instruction(self.system_instruction.as_ref().cloned())
            .contents(messages)
            .build();

        // Update history with user message
        self.history.push(user_message);

        // Start streaming
        self.model.stream_generate_response(request).await
    }

    /// Clears the chat history while keeping the system instruction.
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Returns the current chat history.
    pub fn history(&self) -> &[Content] {
        &self.history
    }

    /// Returns the system instruction if set.
    pub fn system_instruction(&self) -> Option<&SystemInstruction> {
        self.system_instruction.as_ref()
    }
}
