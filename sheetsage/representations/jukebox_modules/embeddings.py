"""
Jukebox Embeddings Extractor - SheetSage Pattern Implementation

This module implements the two-stage embedding extraction pattern used by SheetSage:
1. VQ-VAE Encoding: Audio → discrete codes
2. Language Model Activations: Codes → high-level embeddings

Based on the pattern described in JUKEBOX_EMBEDDINGS_USAGE_2025-11-19.md
"""

import numpy as np
import torch as t
from jukebox_modules.hparams import Hyperparams
from jukebox_modules.make_models import make_model
from jukebox_modules.utils.audio_utils import load_audio
from jukebox_modules.utils.torch_utils import empty_cache

# Constants from SheetSage pattern
_SAMPLE_RATE = 44100  # Hz
_FRAME_HOP_SIZE = 128  # samples (calculated from raw_to_tokens for top level)
_CHUNK_SAMPLES = 1_048_576  # samples (8192 frames × 128 hop ≈ 23.75s at 44.1kHz)
_CHUNK_FRAMES = 8192  # frames (prior.n_ctx)


class JukeboxEmbeddings:
    """
    Jukebox neural embeddings extractor following the SheetSage pattern.

    This class implements the two-stage embedding extraction process:
    1. Audio → VQ-VAE codes (codify_audio)
    2. Codes → LM activations (lm_activations)

    Usage:
        >>> extractor = JukeboxEmbeddings(model_name="5b", device="cuda")
        >>> rate, embeddings = extractor("audio.wav", offset=0.0, duration=30.0)
        >>> # rate = 344.5 Hz
        >>> # embeddings shape: (N, 4800) where N = number of frames
    """

    def __init__(
        self,
        model_name: str = "5b",
        device: str | t.device = "cuda",
        sample_length_in_seconds: float = 24.0,
        num_layers: int | None = None,
        auto_download: bool = True,
    ):
        """
        Initialize Jukebox embeddings extractor.

        Args:
            model_name: Model to use ("5b", "5b_lyrics", or "1b_lyrics")
            device: Device to run on ("cuda" or "cpu")
            sample_length_in_seconds: Max chunk duration in seconds (default 24s)
            num_layers: Which layer to extract activations from (None = deepest/last)
            auto_download: If True, automatically download missing checkpoints
        """
        self.model_name = model_name
        self.device = device if isinstance(device, t.device) else t.device(device)
        self.auto_download = auto_download

        # Initialize models
        hps = Hyperparams(
            sample_length_in_seconds=sample_length_in_seconds,
            total_sample_length_in_seconds=sample_length_in_seconds,
            sr=_SAMPLE_RATE,
            n_samples=1,
        )

        self.vqvae, self.priors = make_model(
            model_name,
            self.device,
            hps,
            levels=None,  # Load all levels
            auto_download=auto_download,
        )

        # Use top-level prior (deepest layer) for embeddings
        self.prior = self.priors[-1]  # Level 2 for 5B model

        # Calculate frame rate from raw_to_tokens
        self.raw_to_tokens = self.prior.raw_to_tokens
        self.frame_rate = _SAMPLE_RATE / self.raw_to_tokens  # ~344.5 Hz

        # Chunk sizes
        self.chunk_samples = _CHUNK_FRAMES * self.raw_to_tokens  # ~1,048,576 samples
        self.chunk_frames = self.prior.n_ctx  # 8192 frames (prior context window)

        # Layer to extract activations from
        if num_layers is None:
            # Extract from deepest layer (last layer)
            # Access protected member _attn_mods to get layer count
            self.num_layers = len(self.prior.prior.transformer._attn_mods) - 1  # noqa: SLF001
        else:
            self.num_layers = num_layers

        # Model width (embedding dimension)
        self.embedding_dim = self.prior.prior.width  # 4800 for 5B model

    def decode_audio(
        self, audio_path: str | bytes, offset: float = 0.0, duration: float | None = None
    ) -> np.ndarray:
        """
        Decode and preprocess audio.

        Args:
            audio_path: Path to audio file
            offset: Start offset in seconds
            duration: Duration in seconds (None = entire file)

        Returns:
            Audio array: (samples, channels) format, mono, normalized
        """
        # Handle bytes input (e.g., from HTTP)
        if isinstance(audio_path, bytes):
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_path)
                temp_path = f.name
            try:
                audio = load_audio(
                    temp_path, sr=_SAMPLE_RATE, offset=offset, duration=duration, mono=True
                )
            finally:
                os.unlink(temp_path)
        else:
            audio = load_audio(
                audio_path, sr=_SAMPLE_RATE, offset=offset, duration=duration, mono=True
            )

        # Return as (samples, channels) - mono is (samples, 1)
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)

        return audio

    def codify_audio(self, audio: np.ndarray, tqdm=lambda x: x) -> t.Tensor:
        """
        Convert raw audio into discrete VQ-VAE codes.

        Process:
        1. Audio is chunked into windows of _CHUNK_SAMPLES (1,048,576 samples ≈ 23.75s at 44.1kHz)
        2. Each chunk is encoded through VQ-VAE encoder
        3. Output: discrete codes at frame rate of 44100/128 = 344.5 Hz

        Args:
            audio: Audio array, shape (samples, channels)
            tqdm: Progress bar function (optional)

        Returns:
            Codes tensor: shape (batch_size, num_frames) where num_frames = samples / raw_to_tokens
        """
        # Convert to tensor and add batch dimension
        # VQVAE expects (batch, time, channels) format
        audio_tensor = t.from_numpy(audio).unsqueeze(0).float().to(self.device)  # (1, T, C)

        total_samples = audio_tensor.shape[1]

        # Chunk audio if necessary
        if total_samples > self.chunk_samples:
            codes_list = []
            for start in tqdm(range(0, total_samples, self.chunk_samples)):
                end = min(start + self.chunk_samples, total_samples)
                chunk = audio_tensor[:, start:end, :]

                # Encode to top-level codes (level 2)
                # Returns list: [level0_codes, level1_codes, level2_codes]
                chunk_codes = self.vqvae.encode(chunk, start_level=2, end_level=3)

                # Get top-level codes only
                codes_list.append(chunk_codes[0])  # shape: (1, T_codes)

                empty_cache()

            # Concatenate codes from all chunks
            codes = t.cat(codes_list, dim=1)  # (1, total_frames)
        else:
            # Single chunk - no chunking needed
            chunk_codes = self.vqvae.encode(audio_tensor, start_level=2, end_level=3)
            codes = chunk_codes[0]  # (1, T_codes)

        return codes

    def lm_activations(
        self,
        audio_codified: t.Tensor,
        metadata_offset_seconds: float = 0.0,
        metadata_total_length_seconds: float | None = None,
        tqdm=lambda x: x,
    ) -> t.Tensor:
        """
        Extract neural activations from Jukebox's language model.

        Process:
        1. VQ-VAE codes are chunked into windows of _CHUNK_FRAMES (8192 frames)
        2. Each chunk is passed through the prior (language model) with conditioning
        3. Activations from the deepest layer are extracted
        4. Output: 4800-dimensional embeddings at 344.5 Hz

        Args:
            audio_codified: VQ-VAE codes, shape (batch_size, num_frames)
            metadata_offset_seconds: Offset in seconds for metadata conditioning
            metadata_total_length_seconds: Total audio length in seconds (optional)
            tqdm: Progress bar function (optional)

        Returns:
            Activations tensor: shape (batch_size, num_frames, embedding_dim)
        """
        codes = audio_codified
        batch_size = codes.shape[0]
        total_frames = codes.shape[1]

        # Prepare metadata labels (optional conditioning)
        y = self._get_labels(metadata_offset_seconds, metadata_total_length_seconds)

        activations_list = []

        # Set up forward hook to capture activations from target layer
        captured_activations = {}

        def create_hook(name):
            def hook_fn(module, input_tensor, output):
                # Output from ResAttnBlock is the hidden state
                # Ignore unused args for standard hook signature
                _ = module  # noqa: F841
                _ = input_tensor  # noqa: F841
                captured_activations[name] = output.detach()

            return hook_fn

        # Register hook on target layer
        # Access protected member _attn_mods to register hooks
        target_layer = self.prior.prior.transformer._attn_mods[self.num_layers]  # noqa: SLF001
        handle = target_layer.register_forward_hook(create_hook("activations"))

        try:
            # Process codes in chunks
            for start in tqdm(range(0, total_frames, self.chunk_frames)):
                end = min(start + self.chunk_frames, total_frames)
                z_chunk = codes[:, start:end]  # (batch_size, chunk_frames)

                # Prepare conditioning
                z_conds = []  # No upper-level conditioning for top-level prior

                # Adjust metadata offset for this chunk
                if y is not None:
                    chunk_y = y.clone()
                    chunk_offset = metadata_offset_seconds + (start / self.frame_rate)
                    # Update offset in labels (y[:, 1] is offset)
                    chunk_y[:, 1:2] = int(chunk_offset * _SAMPLE_RATE)
                    chunk_y = self.prior.get_y([chunk_y], start=0)
                else:
                    chunk_y = None

                # Forward pass through prior
                # This will trigger the hook and capture activations
                with t.no_grad():
                    self.prior.z_forward(z_chunk, z_conds, chunk_y, fp16=True)

                # Get captured activations
                if "activations" in captured_activations:
                    chunk_activations = captured_activations["activations"]
                    # chunk_activations shape: (batch_size, chunk_frames, width)
                    activations_list.append(chunk_activations)
                    captured_activations.clear()

                empty_cache()

            # Concatenate activations from all chunks
            if activations_list:
                activations = t.cat(activations_list, dim=1)  # (batch_size, total_frames, width)
            else:
                # Fallback: return zeros if no activations captured
                activations = t.zeros(
                    (batch_size, total_frames, self.embedding_dim),
                    device=self.device,
                    dtype=t.float32,
                )

        finally:
            # Remove hook
            handle.remove()

        return activations

    def _get_labels(
        self, offset_seconds: float, total_length_seconds: float | None
    ) -> t.Tensor | None:
        """
        Create metadata labels for conditioning.

        Args:
            offset_seconds: Audio offset in seconds
            total_length_seconds: Total audio length in seconds

        Returns:
            Labels tensor or None if not using conditioning
        """
        if not self.prior.y_cond:
            return None

        # Create empty labels (no artist/genre/lyrics conditioning)
        # For embeddings extraction, we typically don't need metadata
        metas = [
            {
                "artist": "",
                "genre": "",
                "lyrics": "",
                "total_length": int((total_length_seconds or 0) * _SAMPLE_RATE),
                "offset": int(offset_seconds * _SAMPLE_RATE),
            }
        ]

        labels = self.prior.labeller.get_batch_labels(metas, self.device)
        return labels

    def __call__(
        self, audio_path: str | bytes, offset: float = 0.0, duration: float | None = None
    ) -> tuple[float, np.ndarray]:
        """
        Main interface: audio file → embeddings.

        This method implements the SheetSage pattern:
        1. Decode and preprocess audio
        2. Encode to VQ-VAE codes
        3. Extract LM activations
        4. Trim to match audio length
        5. Return (rate, activations)

        Args:
            audio_path: Path to audio file or bytes
            offset: Start offset in seconds
            duration: Duration in seconds (None = entire file)

        Returns:
            rate: Frame rate in Hz (344.5 for Jukebox)
            activations: np.ndarray, shape (N, embedding_dim) where N = number of frames
        """
        # 1. Decode and preprocess audio
        audio = self.decode_audio(audio_path, offset=offset, duration=duration)
        total_length_seconds = audio.shape[0] / _SAMPLE_RATE

        # 2. Encode to VQ-VAE codes
        codified_audio = self.codify_audio(audio)

        # 3. Extract LM activations
        activations = self.lm_activations(
            codified_audio,
            metadata_offset_seconds=offset,
            metadata_total_length_seconds=total_length_seconds,
        )

        # 4. Trim to match audio length
        # Activations are at frame_rate Hz, audio is at sample_rate Hz
        target_frames = int(audio.shape[0] / self.raw_to_tokens)
        activations = activations[0, :target_frames, :]  # Remove batch dim: (N, embedding_dim)

        # Convert to numpy
        activations_np = activations.cpu().numpy()

        rate = self.frame_rate  # ~344.5 Hz
        return rate, activations_np


# Convenience function for singleton pattern (as used in SheetSage)
_jukebox_singleton: JukeboxEmbeddings | None = None


def init_jukebox_singleton(
    model_name: str = "5b",
    device: str | t.device = "cuda",
    num_layers: int | None = None,
    auto_download: bool = True,
) -> JukeboxEmbeddings:
    """
    Initialize a singleton Jukebox embeddings extractor.

    This follows the SheetSage pattern where a single model instance
    is reused across multiple calls.

    Args:
        model_name: Model to use ("5b", "5b_lyrics", or "1b_lyrics")
        device: Device to run on ("cuda" or "cpu")
        num_layers: Which layer to extract activations from (None = deepest)
        auto_download: If True, automatically download missing checkpoints

    Returns:
        JukeboxEmbeddings instance (singleton)
    """
    global _jukebox_singleton  # noqa: PLW0603

    if _jukebox_singleton is None:
        _jukebox_singleton = JukeboxEmbeddings(
            model_name=model_name, device=device, num_layers=num_layers, auto_download=auto_download
        )

    return _jukebox_singleton


def get_jukebox_singleton() -> JukeboxEmbeddings | None:
    """Get the current Jukebox singleton instance."""
    return _jukebox_singleton
