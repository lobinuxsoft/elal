use thiserror::Error;

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("failed to load model: {0}")]
    Load(String),

    #[error("failed to create context: {0}")]
    Context(String),

    #[error("tokenization failed: {0}")]
    Tokenize(String),

    #[error("decode failed: {0}")]
    Decode(String),

    #[error("sampling failed: {0}")]
    Sampling(String),

    #[error("chat template application failed: {0}")]
    ChatTemplate(String),

    #[error("channel closed before stream completed")]
    ChannelClosed,

    #[error("llama.cpp backend error: {0}")]
    Llama(String),

    #[error("serialization error: {0}")]
    Serialize(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}
