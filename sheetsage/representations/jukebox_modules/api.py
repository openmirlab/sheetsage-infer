"""
Simple API for Jukebox inference.
"""

import os
import torch
import numpy as np
from jukebox_modules.hparams import Hyperparams
from jukebox_modules.make_models import make_model, download_checkpoints
from jukebox_modules.sample import ancestral_sample, primed_sample, load_prompts
from jukebox_modules.utils.audio_utils import save_wav, load_audio


class Jukebox:
    """
    Simple API for Jukebox music generation.

    Example:
        >>> model = Jukebox("5b_lyrics", device="cuda")
        >>> audio = model.generate(artist="The Beatles", genre="Rock", duration=20)
    """

    def __init__(self, model_name="5b_lyrics", device="cuda"):
        """
        Initialize Jukebox model.

        Args:
            model_name: Model to use ("1b_lyrics", "5b", or "5b_lyrics")
            device: Device to run on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.vqvae = None
        self.priors = None

    def load(self, sample_length_in_seconds=20, n_samples=1, auto_download=True):
        """
        Load the model (downloads checkpoints automatically if needed).

        Args:
            sample_length_in_seconds: Length of audio to generate
            n_samples: Number of samples to generate in parallel
            auto_download: If True, automatically download missing checkpoints
        """
        hps = Hyperparams(
            sample_length_in_seconds=sample_length_in_seconds,
            total_sample_length_in_seconds=sample_length_in_seconds,
            sr=44100,
            n_samples=n_samples,
            hop_fraction=[0.5, 0.5, 0.125]
        )

        print(f"Loading {self.model_name}...")
        if auto_download:
            print("Note: Missing checkpoints will be downloaded automatically.")
        self.vqvae, self.priors = make_model(self.model_name, self.device, hps, auto_download=auto_download)
        self.hps = hps
        print("✓ Model loaded successfully")

    def generate(
        self,
        artist="",
        genre="",
        lyrics="",
        duration_seconds=20,
        temperature=0.99,
        output_path=None
    ):
        """
        Generate music from scratch.

        Args:
            artist: Artist name to condition on
            genre: Genre to condition on
            lyrics: Lyrics to condition on (for lyrics models)
            duration_seconds: Duration of audio to generate
            temperature: Sampling temperature (default 0.99)
            output_path: Where to save audio (optional)

        Returns:
            numpy array of audio samples
        """
        if self.vqvae is None:
            self.load(sample_length_in_seconds=duration_seconds)

        # Create labels
        metas = [dict(
            artist=artist,
            genre=genre,
            lyrics=lyrics,
            total_length=duration_seconds * self.hps.sr,
            offset=0
        )]
        labels = [prior.labeller.get_batch_labels(metas, self.device)
                 for prior in self.priors]

        # Sampling kwargs - optimized for GPU
        # Larger batch sizes = better GPU utilization
        chunk_size = 32 if self.model_name == '1b_lyrics' else 16
        max_batch_size = 32 if self.model_name == '1b_lyrics' else 16
        sampling_kwargs = [
            dict(temp=temperature, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=temperature, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=temperature, fp16=True, chunk_size=chunk_size, max_batch_size=max_batch_size)
        ]

        # Generate
        print("Generating music...")
        # Convert device string to torch.device if needed
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        with torch.no_grad():
            zs = ancestral_sample(labels, sampling_kwargs, self.priors, self.hps, device=device)

        # Decode to audio
        # zs is a list with one tensor per level, decode needs all levels from start_level to end_level
        print("Decoding to audio...")
        audio = self.priors[-1].decode(zs, start_level=0, bs_chunks=1)
        audio = audio.cpu().numpy()

        if output_path:
            # Handle both file paths and directory paths
            if os.path.splitext(output_path)[1]:  # Has extension, treat as file
                output_dir = os.path.dirname(output_path) or "."
                os.makedirs(output_dir, exist_ok=True)
                # Save directly using soundfile for single file
                import soundfile
                aud_clipped = torch.clamp(torch.from_numpy(audio), -1, 1).numpy()
                soundfile.write(output_path, aud_clipped[0], samplerate=self.hps.sr, format='wav')
                print(f"Saved to {output_path}")
            else:  # Directory path
                os.makedirs(output_path, exist_ok=True)
                save_wav(output_path, audio, self.hps.sr)
                print(f"Saved to {output_path}/item_0.wav")

        return audio

    def generate_from_audio(
        self,
        prompt_audio,
        prompt_duration=12,
        total_duration=30,
        output_path=None
    ):
        """
        Generate music continuation from an audio prompt.

        Args:
            prompt_audio: Path to audio file or numpy array
            prompt_duration: How many seconds of prompt to use
            total_duration: Total duration to generate
            output_path: Where to save audio (optional)

        Returns:
            numpy array of audio samples
        """
        if self.vqvae is None:
            self.load(sample_length_in_seconds=total_duration)

        # Load prompt
        if isinstance(prompt_audio, str):
            x = load_prompts([prompt_audio],
                           prompt_duration * self.hps.sr,
                           self.hps,
                           device=self.device)
        else:
            x = torch.from_numpy(prompt_audio).unsqueeze(0).to(self.device)

        # Create empty labels
        metas = [dict(artist="", genre="", lyrics="",
                     total_length=total_duration * self.hps.sr, offset=0)]
        labels = [prior.labeller.get_batch_labels(metas, self.device)
                 for prior in self.priors]

        # Sampling kwargs - optimized for GPU
        chunk_size = 32 if self.model_name == '1b_lyrics' else 16
        max_batch_size = 32 if self.model_name == '1b_lyrics' else 16
        sampling_kwargs = [
            dict(temp=0.99, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=0.99, fp16=True, chunk_size=64, max_batch_size=32),
            dict(temp=0.99, fp16=True, chunk_size=chunk_size, max_batch_size=max_batch_size)
        ]

        # Generate
        print("Generating continuation...")
        # Convert device string to torch.device if needed
        device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        with torch.no_grad():
            zs = primed_sample(x, labels, sampling_kwargs, self.priors, self.hps, device=device)

        # Decode to audio
        # zs is a list with one tensor per level, decode needs all levels from start_level to end_level
        print("Decoding to audio...")
        audio = self.priors[-1].decode(zs, start_level=0, bs_chunks=1)
        audio = audio.cpu().numpy()

        if output_path:
            # Handle both file paths and directory paths
            if os.path.splitext(output_path)[1]:  # Has extension, treat as file
                output_dir = os.path.dirname(output_path) or "."
                os.makedirs(output_dir, exist_ok=True)
                # Save directly using soundfile for single file
                import soundfile
                aud_clipped = torch.clamp(torch.from_numpy(audio), -1, 1).numpy()
                soundfile.write(output_path, aud_clipped[0], samplerate=self.hps.sr, format='wav')
                print(f"Saved to {output_path}")
            else:  # Directory path
                os.makedirs(output_path, exist_ok=True)
                save_wav(output_path, audio, self.hps.sr)
                print(f"Saved to {output_path}/item_0.wav")

        return audio
