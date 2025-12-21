import logging
import os
import sys
import warnings

import librosa
import numpy as np
import torch as t

from ..utils import get_approximate_audio_length

# Add jukebox_modules to path if it exists locally
_jukebox_modules_path = os.path.join(os.path.dirname(__file__), "jukebox_modules")
if os.path.exists(_jukebox_modules_path) and _jukebox_modules_path not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from jukebox_modules.hparams import Hyperparams
from jukebox_modules.make_models import make_model
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

        def _build_models(target_device):
            return make_model(
                model_name,
                target_device,
                hps,
                levels=None,  # Load all levels
                auto_download=auto_download,
            )

        try:
            self.vqvae, self.priors = _build_models(self.device)
        except RuntimeError as e:
            if (
                "out of memory" in str(e).lower()
                and isinstance(self.device, t.device)
                and self.device.type == "cuda"
            ):
                logging.warning(
                    "CUDA out of memory while loading Jukebox (device=%s). "
                    "Falling back to CPU – this will be much slower.",
                    self.device,
                )
                self.device = t.device("cpu")
                self.vqvae, self.priors = _build_models(self.device)
            else:
                raise

        # Use top-level prior (deepest layer) for embeddings
        self.prior = self.priors[-1]  # Level 2 for 5B model

        # Calculate frame rate from raw_to_tokens
        # Try to access raw_to_tokens from prior
        try:
            self.raw_to_tokens = self.prior.raw_to_tokens
        except AttributeError:
            # Try alternative path
            self.raw_to_tokens = self.prior.prior.raw_to_tokens
        self.frame_rate = _SAMPLE_RATE / self.raw_to_tokens  # ~344.5 Hz

        # Chunk sizes
        self.chunk_frames = int(self.prior.n_ctx)  # Prior context window
        self.chunk_samples = int(self.chunk_frames * self.raw_to_tokens)

        # Layer to extract activations from
        num_attn_layers = len(self.prior.prior.transformer._attn_mods)  # noqa: SLF001
        if num_layers is None:
            # Extract from deepest layer (last layer)
            self.num_layers = num_attn_layers - 1
        else:
            if num_layers >= num_attn_layers:
                raise ValueError(
                    f"num_layers {num_layers} >= {num_attn_layers} available layers. "
                    f"Use a value between 0 and {num_attn_layers - 1}"
                )
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
            Audio array: (samples,) format, mono, normalized
        """
        # Handle bytes input (e.g., from HTTP)
        if isinstance(audio_path, bytes):
            import os
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_path)
                temp_path = f.name
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio, sr = librosa.load(
                        temp_path, sr=None, mono=False, offset=offset, duration=duration
                    )
            finally:
                os.unlink(temp_path)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio, sr = librosa.load(
                    audio_path, sr=None, mono=False, offset=offset, duration=duration
                )

        # Convert to mono if stereo
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]
        audio = np.swapaxes(audio, 0, 1)  # (channels, samples) -> (samples, channels)
        audio = np.mean(audio, axis=1, keepdims=False)  # Convert to mono: (samples,)

        # Resample to target sample rate if needed
        if sr != _SAMPLE_RATE:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=_SAMPLE_RATE, res_type="kaiser_best"
            )

        # Normalize
        if audio.shape[0] > 0:
            norm_factor = np.abs(audio).max()
            if norm_factor > 0:
                audio /= norm_factor

        return audio

    def codify_audio(self, audio: np.ndarray, tqdm=lambda x: x) -> t.Tensor:
        """
        Convert raw audio into discrete VQ-VAE codes using model-specific chunk sizes.
        """
        hop_size = self.chunk_samples
        hop_size_frames = hop_size // _FRAME_HOP_SIZE
        result = []
        for start in tqdm(range(0, audio.shape[0], hop_size)):
            context = audio[start : start + hop_size]
            if context.shape[0] < hop_size:
                context = np.pad(context, (0, hop_size - context.shape[0]))
            with t.no_grad():
                context_tensor = t.from_numpy(context).float().to(self.device).view(1, -1, 1)
                context_codified = self.vqvae.encode(context_tensor, start_level=2, end_level=3)[
                    -1
                ].view(-1)
            context_codified = context_codified[:hop_size_frames]
            result.append(context_codified)
        codes = t.cat(result, dim=0)
        codes = codes.view(1, -1)
        return codes

    def lm_activations(
        self,
        audio_codified: t.Tensor,
        metadata_offset_seconds: float = 0.0,
        metadata_total_length_seconds: float | None = None,
        chunk_duration_seconds: float | None = None,
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
        total_frames = int(codes.shape[1])

        # Prepare metadata labels (optional conditioning)
        y = self._get_labels(
            metadata_offset_seconds,
            metadata_total_length_seconds,
            chunk_duration_seconds,
        )

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
                    # y is a dict with 'y' and 'info' keys from get_batch_labels
                    chunk_offset = metadata_offset_seconds + (start / self.frame_rate)
                    # Create new labels dict for this chunk
                    chunk_labels = {
                        "y": y["y"].clone(),
                        "info": y["info"],  # Copy info reference
                    }
                    # Update offset in y tensor (y[:, 1] is offset)
                    chunk_labels["y"][:, 1:2] = int(chunk_offset * _SAMPLE_RATE)
                    # Pass dict to get_y (not a list)
                    chunk_y = self.prior.get_y(chunk_labels, start=0)
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
        self,
        offset_seconds: float,
        total_length_seconds: float | None,
        chunk_duration_seconds: float | None,
    ) -> dict | None:
        """
        Create metadata labels for conditioning.

        Args:
            offset_seconds: Audio offset in seconds
            total_length_seconds: Total audio length in seconds

        Returns:
            Labels dict with 'y' and 'info' keys, or None if not using conditioning
        """
        if not self.prior.y_cond:
            return None

        # Ensure total length is at least offset + chunk duration
        effective_total_length_seconds = total_length_seconds
        if effective_total_length_seconds is None or effective_total_length_seconds <= 0:
            effective_total_length_seconds = chunk_duration_seconds
        if effective_total_length_seconds is None:
            effective_total_length_seconds = 0.0
        if effective_total_length_seconds <= offset_seconds:
            fallback = chunk_duration_seconds or (
                self.chunk_frames * self.raw_to_tokens / _SAMPLE_RATE
            )
            effective_total_length_seconds = offset_seconds + fallback

        # Create empty labels (no artist/genre/lyrics conditioning)
        # For embeddings extraction, we typically don't need metadata
        metas = [
            {
                "artist": "",
                "genre": "",
                "lyrics": "",
                "total_length": int(effective_total_length_seconds * _SAMPLE_RATE),
                "offset": int(offset_seconds * _SAMPLE_RATE),
            }
        ]

        # get_batch_labels returns dict with 'y' (tensor) and 'info' (list) keys
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
        chunk_duration_seconds = audio.shape[0] / _SAMPLE_RATE

        if offset == 0.0 and duration is None:
            total_length_seconds = chunk_duration_seconds
        else:
            try:
                total_length_seconds = get_approximate_audio_length(audio_path)
            except Exception as exc:
                logging.warning(
                    "Failed to estimate total audio length via ffprobe (%s). "
                    "Falling back to chunk duration.",
                    exc,
                )
                total_length_seconds = chunk_duration_seconds + offset

        # 2. Encode to VQ-VAE codes
        codified_audio = self.codify_audio(audio)

        # 3. Extract LM activations
        activations = self.lm_activations(
            codified_audio,
            metadata_offset_seconds=offset,
            metadata_total_length_seconds=total_length_seconds,
            chunk_duration_seconds=chunk_duration_seconds,
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
