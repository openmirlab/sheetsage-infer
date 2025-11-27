import torch as t

from jukebox_modules.hparams import Hyperparams

# Import torch.cuda for device checks
if not hasattr(t, 'cuda'):
    # Fallback if torch.cuda not available
    class CudaStub:
        @staticmethod
        def is_available():
            return False
    t.cuda = CudaStub()
from jukebox_modules.utils.torch_utils import empty_cache
from jukebox_modules.utils.audio_utils import load_audio
from jukebox_modules.utils.sample_utils import split_batch, get_starts
from jukebox_modules.utils.dist_utils import print_once

# Sample a partial window of length<n_ctx with tokens_to_sample new tokens on level=level
def sample_partial_window(zs, labels, sampling_kwargs, level, prior, tokens_to_sample, hps):
    z = zs[level]
    n_ctx = prior.n_ctx
    current_tokens = z.shape[1]
    if current_tokens < n_ctx - tokens_to_sample:
        sampling_kwargs['sample_tokens'] = current_tokens + tokens_to_sample
        start = 0
    else:
        sampling_kwargs['sample_tokens'] = n_ctx
        start = current_tokens - n_ctx + tokens_to_sample

    return sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)

# Sample a single window of length=n_ctx at position=start on level=level
def sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps):
    n_samples = hps.n_samples
    n_ctx = prior.n_ctx
    end = start + n_ctx

    # get z already sampled at current level
    z = zs[level][:,start:end]

    if 'sample_tokens' in sampling_kwargs:
        # Support sampling a window shorter than n_ctx
        sample_tokens = sampling_kwargs['sample_tokens']
    else:
        sample_tokens = (end - start)
    conditioning_tokens, new_tokens = z.shape[1], sample_tokens - z.shape[1]

    print_once(f"Sampling {sample_tokens} tokens for [{start},{start+sample_tokens}]. Conditioning on {conditioning_tokens} tokens")

    if new_tokens <= 0:
        # Nothing new to sample
        return zs
    
    # get z_conds from level above
    z_conds = prior.get_z_conds(zs, start, end)

    # set y offset, sample_length and lyrics tokens
    y = prior.get_y(labels, start)

    empty_cache()

    max_batch_size = sampling_kwargs['max_batch_size']
    del sampling_kwargs['max_batch_size']


    z_list = split_batch(z, n_samples, max_batch_size)
    z_conds_list = split_batch(z_conds, n_samples, max_batch_size)
    y_list = split_batch(y, n_samples, max_batch_size)
    z_samples = []
    for z_i, z_conds_i, y_i in zip(z_list, z_conds_list, y_list):
        z_samples_i = prior.sample(n_samples=z_i.shape[0], z=z_i, z_conds=z_conds_i, y=y_i, **sampling_kwargs)
        z_samples.append(z_samples_i)
    z = t.cat(z_samples, dim=0)

    sampling_kwargs['max_batch_size'] = max_batch_size

    # Update z with new sample
    z_new = z[:,-new_tokens:]
    zs[level] = t.cat([zs[level], z_new], dim=1)
    return zs

# Sample total_length tokens at level=level with hop_length=hop_length
def sample_level(zs, labels, sampling_kwargs, level, prior, total_length, hop_length, hps):
    print_once(f"Sampling level {level}")
    if total_length >= prior.n_ctx:
        for start in get_starts(total_length, prior.n_ctx, hop_length):
            zs = sample_single_window(zs, labels, sampling_kwargs, level, prior, start, hps)
    else:
        zs = sample_partial_window(zs, labels, sampling_kwargs, level, prior, total_length, hps)
    return zs

# Sample multiple levels
def _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps, device=None):
    # Get device from first prior if not provided (all priors should be on same device)
    # Check the device the prior SHOULD be on, not where it currently is
    # (it might be on CPU from previous level for memory management)
    if device is None:
        if len(priors) > 0:
            # Try to get device from model parameters, but if it's on CPU, 
            # check the first parameter's device type to determine target device
            try:
                first_param_device = next(priors[0].parameters()).device
                # If it's on CPU, we still want to use GPU if available
                # Check if CUDA is available and use it
                if first_param_device.type == 'cpu' and t.cuda.is_available():
                    device = t.device('cuda')
                else:
                    device = first_param_device
            except StopIteration:
                device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        else:
            device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    
    # Ensure device is a torch.device object
    if isinstance(device, str):
        device = t.device(device)
    
    for level in reversed(sample_levels):
        prior = priors[level]
        # Move prior to device (respects device from model loading)
        prior.to(device)
        empty_cache()

        # Set correct total_length, hop_length, labels and sampling_kwargs for level
        assert hps.sample_length % prior.raw_to_tokens == 0, f"Expected sample_length {hps.sample_length} to be multiple of {prior.raw_to_tokens}"
        total_length = hps.sample_length//prior.raw_to_tokens
        hop_length = int(hps.hop_fraction[level]*prior.n_ctx)
        zs = sample_level(zs, labels[level], sampling_kwargs[level], level, prior, total_length, hop_length, hps)

        # Move prior to CPU to save memory (only if not already on CPU)
        if device.type != 'cpu':
            prior.cpu()
        empty_cache()
    return zs

# Generate ancestral samples given a list of artists and genres
def ancestral_sample(labels, sampling_kwargs, priors, hps, device=None):
    sample_levels = list(range(len(priors)))
    # Determine device - prefer provided device, otherwise infer from priors or CUDA
    if device is None:
        if len(priors) > 0:
            try:
                device = next(priors[0].parameters()).device
                # If on CPU but CUDA available, use CUDA
                if device.type == 'cpu' and t.cuda.is_available():
                    device = t.device('cuda')
            except StopIteration:
                device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
        else:
            device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    if isinstance(device, str):
        device = t.device(device)
    zs = [t.zeros(hps.n_samples,0,dtype=t.long, device=device) for _ in range(len(priors))]
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps, device=device)
    return zs

# Continue ancestral sampling from previously saved codes
def continue_sample(zs, labels, sampling_kwargs, priors, hps, device=None):
    sample_levels = list(range(len(priors)))
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps, device=device)
    return zs

# Upsample given already generated upper-level codes
def upsample(zs, labels, sampling_kwargs, priors, hps, device=None):
    sample_levels = list(range(len(priors) - 1))
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps, device=device)
    return zs

# Prompt the model with raw audio input (dimension: NTC) and generate continuations
def primed_sample(x, labels, sampling_kwargs, priors, hps, device=None):
    sample_levels = list(range(len(priors)))
    zs = priors[-1].encode(x, start_level=0, end_level=len(priors), bs_chunks=x.shape[0])
    zs = _sample(zs, labels, sampling_kwargs, priors, sample_levels, hps, device=device)
    return zs

# Load `duration` seconds of the given audio files to use as prompts
def load_prompts(audio_files, duration, hps, device='cuda'):
    xs = []
    for audio_file in audio_files:
        x = load_audio(audio_file, sr=hps.sr, duration=duration, offset=0.0, mono=True)
        x = x.T # CT -> TC
        xs.append(x)
    while len(xs) < hps.n_samples:
        xs.extend(xs)
    xs = xs[:hps.n_samples]
    x = t.stack([t.from_numpy(x) for x in xs])
    x = x.to(device, non_blocking=True)
    return x

# Load codes from previous sampling run
def load_codes(codes_file, duration, priors, hps, device='cuda'):
    data = t.load(codes_file, map_location='cpu')
    # Get device from first prior if available, otherwise use provided device
    if len(priors) > 0:
        prior_device = next(priors[0].parameters()).device
        device = prior_device if device == 'cuda' else device
    zs = [z.to(device) for z in data['zs']]
    assert zs[-1].shape[0] == hps.n_samples, f"Expected bs = {hps.n_samples}, got {zs[-1].shape[0]}"
    del data
    if duration is not None:
        # Cut off codes to match duration
        top_raw_to_tokens = priors[-1].raw_to_tokens
        assert duration % top_raw_to_tokens == 0, f"Cut-off duration {duration} not an exact multiple of top_raw_to_tokens"
        assert duration//top_raw_to_tokens <= zs[-1].shape[1], f"Cut-off tokens {duration//priors[-1].raw_to_tokens} longer than tokens {zs[-1].shape[1]} in saved codes"
        zs = [z[:,:duration//prior.raw_to_tokens] for z, prior in zip(zs, priors)]
    return zs
