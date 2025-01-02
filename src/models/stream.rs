use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures::Stream;

use crate::error::GoogleGenerativeAIError;

use super::Response;

/// A custom stream for generating response
pub struct ResponseStream {
    receiver: tokio::sync::mpsc::Receiver<Result<Response, GoogleGenerativeAIError>>,
}

impl ResponseStream {
    /// Creates a new ContentStream
    pub fn new(
        receiver: tokio::sync::mpsc::Receiver<Result<Response, GoogleGenerativeAIError>>,
    ) -> Self {
        Self { receiver }
    }
}

impl Stream for ResponseStream {
    type Item = Result<Response, GoogleGenerativeAIError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.receiver.poll_recv(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}
