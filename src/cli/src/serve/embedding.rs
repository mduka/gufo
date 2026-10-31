//! `gufo serve embedding`: EmbeddingGemma 2 on the Rust engine behind the
//! embedding service.

use std::path::{Path, PathBuf};

use gufo_core::image::decode_image;
use gufo_core::text::CancellationCheck;
use gufo_embedding_gemma2::audio_decoder::decode_audio;
use gufo_embedding_gemma2::audio_processor::{
    AudioFeatures, AudioProcessError, SequenceAudio, process_audio, validate_audio_processor,
};
use gufo_embedding_gemma2::config;
use gufo_embedding_gemma2::image_processor::{
    ImagePatches, process_image, validate_image_processor,
};
use gufo_embedding_gemma2::{
    EncoderRuntime, EncoderSequence, MAX_SEQUENCE_TOKENS, SequenceImage, Tokenizer,
};
use gufo_serve::embedding_service::{
    self, EmbeddingEngine, EmbeddingService, EmbeddingServiceOptions, EngineMedia, EngineSequence,
    MAX_INPUT_TOKENS, load_sentence_transformer_prompts,
};

use super::ModelLoadLog;

// The service's input limit is the model's context window, and its
// placeholders are the model's.
const _: () = assert!(MAX_INPUT_TOKENS == MAX_SEQUENCE_TOKENS);
const _: () = assert!(embedding_service::IMAGE_TOKEN_ID == config::IMAGE_TOKEN_ID);
const _: () = assert!(embedding_service::AUDIO_TOKEN_ID == config::AUDIO_TOKEN_ID);

/// The tokenizer, the media preprocessing and the GPU encoder, which shares
/// device batches between concurrent requests itself.
struct Engine {
    tokenizer: Tokenizer,
    encoder: EncoderRuntime,
    image_processor_error: String,
    audio_processor_error: String,
}

impl EmbeddingEngine for Engine {
    fn max_token_bytes(&self) -> usize {
        self.tokenizer.max_token_bytes()
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text.as_bytes(), true)
    }

    fn encode_truncated(&self, text: &str, max_tokens: usize) -> Vec<u32> {
        self.tokenizer.encode_truncated(text.as_bytes(), max_tokens)
    }

    fn supports_vision(&self) -> bool {
        self.encoder.config().supported_vision
    }

    fn supports_audio(&self) -> bool {
        self.encoder.config().supported_audio
    }

    fn image_processor_error(&self) -> String {
        self.image_processor_error.clone()
    }

    fn audio_processor_error(&self) -> String {
        self.audio_processor_error.clone()
    }

    fn process_image(&self, bytes: &[u8]) -> Result<EngineMedia, String> {
        let image = decode_image(bytes).map_err(|e| e.message().to_owned())?;
        let patches = process_image(&image)?;
        Ok(EngineMedia {
            soft_tokens: patches.soft_tokens(),
            data: Box::new(patches),
        })
    }

    fn process_audio(
        &self,
        bytes: &[u8],
        is_cancelled: Option<&CancellationCheck>,
    ) -> Result<EngineMedia, String> {
        let cancelled = is_cancelled.map(|check| move || check());
        let poll = cancelled.as_ref().map(|c| c as &dyn Fn() -> bool);
        let waveform = decode_audio(bytes, poll).map_err(|e| e.to_string())?;
        let features = process_audio(&waveform, 0, poll).map_err(|e| match e {
            AudioProcessError::Invalid(message) => message,
            AudioProcessError::Cancelled => e.to_string(),
        })?;
        Ok(EngineMedia {
            soft_tokens: features.soft_tokens(),
            data: Box::new(features),
        })
    }

    fn embed(
        &self,
        sequences: Vec<EngineSequence>,
        is_cancelled: Option<CancellationCheck>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let mut inputs = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            let mut input = EncoderSequence::text(sequence.tokens);
            input.is_cancelled = is_cancelled.clone();
            for (offset, media) in sequence.images {
                let patches = media
                    .data
                    .downcast::<ImagePatches>()
                    .map_err(|_| "embedding image of another engine".to_owned())?;
                input.images.push(SequenceImage {
                    offset,
                    patches: *patches,
                });
            }
            for (offset, media) in sequence.audio {
                let features = media
                    .data
                    .downcast::<AudioFeatures>()
                    .map_err(|_| "embedding audio of another engine".to_owned())?;
                input.audio.push(SequenceAudio {
                    offset,
                    features: *features,
                });
            }
            inputs.push(input);
        }
        self.encoder.embed_multimodal(inputs, is_cancelled)
    }

    fn device_lost(&self) -> bool {
        self.encoder.device_lost()
    }
}

/// Loads the tokenizer, then uploads and tunes the encoder.
fn engine(model_root: &Path) -> Result<Box<dyn EmbeddingEngine>, String> {
    let tokenizer = Tokenizer::load(model_root)
        .map_err(|error| format!("embedding tokenizer failed to load: {error}"))?;
    let image_processor_error = validate_image_processor(model_root)
        .err()
        .unwrap_or_default();
    let audio_processor_error = validate_audio_processor(model_root)
        .err()
        .unwrap_or_default();
    let encoder = EncoderRuntime::create(model_root)
        .map_err(|error| format!("embedding encoder failed to load: {error}"))?;
    Ok(Box::new(Engine {
        tokenizer,
        encoder,
        image_processor_error,
        audio_processor_error,
    }))
}

/// The embedding service of `--model`, or the exit status after printing
/// why it cannot serve.
pub(super) fn load(
    model: &Path,
    name: &str,
    default_prompt_name: &str,
) -> Result<EmbeddingService, u8> {
    let mut load_log = ModelLoadLog::new("embedding_gemma2", model);
    let prompts = load_sentence_transformer_prompts(model).map_err(|error| {
        eprintln!("Error: {error}");
        1
    })?;
    let service = EmbeddingService::new(EmbeddingServiceOptions {
        model_root: PathBuf::from(model),
        validate_model: true,
        model_validator: Some(Box::new(gufo_embedding_gemma2::looks_like_embedding_gemma2)),
        model_id: name.to_owned(),
        prompts,
        default_prompt_name: default_prompt_name.to_owned(),
        runner: None,
        engine_factory: Some(Box::new(engine)),
    });
    if !service.ready() {
        eprintln!(
            "Error enabling EmbeddingGemma 2 service: {}",
            service.initialization_error()
        );
        return Err(1);
    }
    load_log.complete(
        &format!(
            "model={} context_tokens={MAX_INPUT_TOKENS}",
            service.model_id()
        ),
        true,
    );
    Ok(service)
}
