"""
Make model classes
Load from checkpoints
Test on dummy outputs to see if everything matches
"""
import os
import numpy as np
import torch as t
import jukebox_modules.utils.dist_adapter as dist
from jukebox_modules.hparams import Hyperparams, setup_hparams, REMOTE_PREFIX
from jukebox_modules.utils.remote_utils import download, check_file_exists
from jukebox_modules.utils.torch_utils import freeze_model
from jukebox_modules.utils.dist_utils import print_all
from jukebox_modules.vqvae.vqvae import calculate_strides

MODELS = {
    '5b': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_5b"),
    '5b_lyrics': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_5b_lyrics"),
    '1b_lyrics': ("vqvae", "upsampler_level_0", "upsampler_level_1", "prior_1b_lyrics"),
    #'your_model': ("you_vqvae_here", "your_upsampler_here", ..., "you_top_level_prior_here")
}

def load_checkpoint(path, auto_download=True):
    """
    Load a checkpoint from path. If path is a remote URL, download it first.
    
    Args:
        path: Local file path or remote URL starting with REMOTE_PREFIX
        auto_download: If True, automatically download missing checkpoints
    """
    restore = path
    if restore.startswith(REMOTE_PREFIX):
        remote_path = restore
        local_path = os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):])
        
        # Check if file exists, download if missing
        if not check_file_exists(local_path):
            if auto_download:
                if dist.get_rank() == 0:
                    download(remote_path, local_path, show_progress=True)
                dist.barrier()  # Wait for download to complete
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at {local_path}\n"
                    f"Please download it manually or set auto_download=True.\n"
                    f"Remote URL: {remote_path}"
                )
        elif dist.get_rank() == 0:
            print(f"✓ Using cached checkpoint: {local_path}")
        
        restore = local_path
    
    if not os.path.exists(restore):
        raise FileNotFoundError(f"Checkpoint file not found: {restore}")
    
    # Check file size - if too small, it might be corrupted
    file_size = os.path.getsize(restore)
    if file_size < 1024:  # Less than 1KB is definitely corrupted
        if dist.get_rank() == 0:
            print(f"⚠ Checkpoint file appears corrupted (size: {file_size} bytes). Removing and re-downloading...")
        os.remove(restore)
        # Re-download if it was a remote path
        if path.startswith(REMOTE_PREFIX):
            remote_path = path
            local_path = os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):])
            if dist.get_rank() == 0:
                download(remote_path, local_path, show_progress=True)
            dist.barrier()
            restore = local_path
        else:
            raise FileNotFoundError(f"Corrupted checkpoint removed: {restore}. Please re-download manually.")
    
    dist.barrier()
    try:
        checkpoint = t.load(restore, map_location=t.device('cpu'))
        if dist.get_rank() == 0:
            print(f"✓ Loaded checkpoint: {restore}")
        return checkpoint
    except (RuntimeError, EOFError) as e:
        # Handle corrupted checkpoint files
        if dist.get_rank() == 0:
            print(f"⚠ Checkpoint file corrupted: {e}")
            print(f"  Removing corrupted file and attempting re-download...")
        os.remove(restore)
        
        # Re-download if it was a remote path
        if path.startswith(REMOTE_PREFIX):
            remote_path = path
            local_path = os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):])
            if dist.get_rank() == 0:
                print(f"  Re-downloading from {remote_path}...")
                download(remote_path, local_path, show_progress=True)
            dist.barrier()
            restore = local_path
            
            # Try loading again after re-download
            try:
                checkpoint = t.load(restore, map_location=t.device('cpu'))
                if dist.get_rank() == 0:
                    print(f"✓ Successfully loaded re-downloaded checkpoint: {restore}")
                return checkpoint
            except (RuntimeError, EOFError) as e2:
                raise RuntimeError(f"Checkpoint still corrupted after re-download: {e2}. "
                                 f"Please manually download: {remote_path}")
        else:
            raise RuntimeError(f"Checkpoint file corrupted and cannot be re-downloaded (local path): {restore}. "
                             f"Error: {e}")

def restore_model(hps, model, checkpoint_path, auto_download=True):
    model.step = 0
    if checkpoint_path != '':
        checkpoint = load_checkpoint(checkpoint_path, auto_download=auto_download)
        # checkpoint_hps = Hyperparams(**checkpoint['hps'])
        # for k in set(checkpoint_hps.keys()).union(set(hps.keys())):
        #     if checkpoint_hps.get(k, None) != hps.get(k, None):
        #         print(k, "Checkpoint:", checkpoint_hps.get(k, None), "Ours:", hps.get(k, None))
        checkpoint['model'] = {k[7:] if k[:7] == 'module.' else k: v for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'])
        if 'step' in checkpoint: model.step = checkpoint['step']

def make_vqvae(hps, device='cuda', auto_download=True):
    from jukebox_modules.vqvae.vqvae import VQVAE
    block_kwargs = dict(width=hps.width, depth=hps.depth, m_conv=hps.m_conv,
                        dilation_growth_rate=hps.dilation_growth_rate,
                        dilation_cycle=hps.dilation_cycle,
                        reverse_decoder_dilation=hps.vqvae_reverse_decoder_dilation)

    if not hps.sample_length:
        assert hps.sample_length_in_seconds != 0
        downsamples = calculate_strides(hps.strides_t, hps.downs_t)
        top_raw_to_tokens = np.prod(downsamples)
        hps.sample_length = (hps.sample_length_in_seconds * hps.sr // top_raw_to_tokens) * top_raw_to_tokens
        print(f"Setting sample length to {hps.sample_length} (i.e. {hps.sample_length/hps.sr} seconds) to be multiple of {top_raw_to_tokens}")

    vqvae = VQVAE(input_shape=(hps.sample_length,1), levels=hps.levels, downs_t=hps.downs_t, strides_t=hps.strides_t,
                  emb_width=hps.emb_width, l_bins=hps.l_bins,
                  mu=hps.l_mu, commit=hps.commit,
                  spectral=hps.spectral, multispectral=hps.multispectral,
                  multipliers=hps.hvqvae_multipliers, use_bottleneck=hps.use_bottleneck,
                  **block_kwargs)

    vqvae = vqvae.to(device)
    restore_model(hps, vqvae, hps.restore_vqvae, auto_download=auto_download)
    # Inference-only: always use eval mode
    print_all(f"Loading vqvae in eval mode")
    vqvae.eval()
    freeze_model(vqvae)
    return vqvae

def make_prior(hps, vqvae, device='cuda', auto_download=True):
    from jukebox_modules.prior.prior import SimplePrior

    prior_kwargs = dict(input_shape=(hps.n_ctx,), bins=vqvae.l_bins,
                        width=hps.prior_width, depth=hps.prior_depth, heads=hps.heads,
                        attn_order=hps.attn_order, blocks=hps.blocks, spread=hps.spread,
                        attn_dropout=hps.attn_dropout, resid_dropout=hps.resid_dropout, emb_dropout=hps.emb_dropout,
                        zero_out=hps.zero_out, res_scale=hps.res_scale, pos_init=hps.pos_init,
                        init_scale=hps.init_scale,
                        m_attn=hps.m_attn, m_mlp=hps.m_mlp,
                        checkpoint_res=0, checkpoint_attn=0, checkpoint_mlp=0)

    x_cond_kwargs = dict(out_width=hps.prior_width, init_scale=hps.init_scale,
                         width=hps.cond_width, depth=hps.cond_depth, m_conv=hps.cond_m_conv,
                         dilation_growth_rate=hps.cond_dilation_growth_rate, dilation_cycle=hps.cond_dilation_cycle,
                         zero_out=hps.cond_zero_out, res_scale=hps.cond_res_scale,
                         checkpoint_res=hps.cond_c_res)  # have to keep this else names wrong

    y_cond_kwargs = dict(out_width=hps.prior_width, init_scale=hps.init_scale,
                         y_bins=hps.y_bins, t_bins=hps.t_bins, sr= hps.sr, min_duration=hps.min_duration,
                         max_duration=hps.max_duration, max_bow_genre_size=hps.max_bow_genre_size)

    if hps.use_tokens and not hps.single_enc_dec:
        prime_kwargs = dict(use_tokens=hps.use_tokens, prime_loss_fraction=hps.prime_loss_fraction,
                            n_tokens=hps.n_tokens, bins=hps.n_vocab,
                            width=hps.prime_width, depth=hps.prime_depth, heads=hps.prime_heads,
                            attn_order=hps.prime_attn_order, blocks=hps.prime_blocks, spread=hps.prime_spread,
                            attn_dropout=hps.prime_attn_dropout, resid_dropout=hps.prime_resid_dropout,
                            emb_dropout=hps.prime_emb_dropout,
                            zero_out=hps.prime_zero_out, res_scale=hps.prime_res_scale,
                            pos_init=hps.prime_pos_init, init_scale=hps.prime_init_scale,
                            m_attn=hps.prime_m_attn, m_mlp=hps.prime_m_mlp,
                            checkpoint_res=0, checkpoint_attn=0,
                            checkpoint_mlp=0)
    else:
        prime_kwargs = dict(use_tokens=hps.use_tokens, prime_loss_fraction=hps.prime_loss_fraction,
                            n_tokens=hps.n_tokens, bins=hps.n_vocab)

    # z_shapes for other levels given this level gets n_ctx codes
    rescale = lambda z_shape: (z_shape[0]*hps.n_ctx//vqvae.z_shapes[hps.level][0],)
    z_shapes = [rescale(z_shape) for z_shape in vqvae.z_shapes]

    prior = SimplePrior(z_shapes=z_shapes,
                        l_bins=vqvae.l_bins,
                        encoder=vqvae.encode,
                        decoder=vqvae.decode,
                        level=hps.level,
                        downs_t=vqvae.downs_t,
                        strides_t=vqvae.strides_t,
                        labels=hps.labels,
                        prior_kwargs=prior_kwargs,
                        x_cond_kwargs=x_cond_kwargs,
                        y_cond_kwargs=y_cond_kwargs,
                        prime_kwargs=prime_kwargs,
                        copy_input=hps.copy_input,
                        labels_v3=hps.labels_v3,
                        merged_decoder=hps.merged_decoder,
                        single_enc_dec=hps.single_enc_dec)

    prior.alignment_head = hps.get('alignment_head', None)
    prior.alignment_layer = hps.get('alignment_layer', None)

    if hps.fp16_params:
        print_all("Converting to fp16 params")
        from jukebox_modules.transformer.ops import _convert_conv_weights_to_fp16
        prior.apply(_convert_conv_weights_to_fp16)
    prior = prior.to(device)
    restore_model(hps, prior, hps.restore_prior, auto_download=auto_download)
    # Inference-only: always use eval mode
    print_all(f"Loading prior in eval mode")
    prior.eval()
    freeze_model(prior)
    return prior

def download_checkpoints(model_name, show_progress=True):
    """
    Pre-download all checkpoints required for a model.
    
    Args:
        model_name: Model name ('1b_lyrics', '5b', or '5b_lyrics')
        show_progress: If True, show download progress bars
    
    Returns:
        dict: Status of each checkpoint download
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    vqvae_name, *prior_names = MODELS[model_name]
    
    # Get checkpoint paths
    vqvae_hps = setup_hparams(vqvae_name, {})
    prior_hps_list = [setup_hparams(name, {}) for name in prior_names]
    
    results = {}
    
    # Download VQ-VAE
    if vqvae_hps.restore_vqvae.startswith(REMOTE_PREFIX):
        remote_path = vqvae_hps.restore_vqvae
        local_path = os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):])
        try:
            download(remote_path, local_path, show_progress=show_progress)
            results['vqvae'] = 'downloaded'
        except Exception as e:
            results['vqvae'] = f'failed: {e}'
            raise
    
    # Download priors
    for i, prior_hps in enumerate(prior_hps_list):
        if prior_hps.restore_prior.startswith(REMOTE_PREFIX):
            remote_path = prior_hps.restore_prior
            local_path = os.path.join(os.path.expanduser("~/.cache"), remote_path[len(REMOTE_PREFIX):])
            try:
                download(remote_path, local_path, show_progress=show_progress)
                results[f'prior_level_{i}'] = 'downloaded'
            except Exception as e:
                results[f'prior_level_{i}'] = f'failed: {e}'
                raise
    
    return results


def make_model(model, device, hps, levels=None, auto_download=True):
    """
    Create and load a Jukebox model.
    
    Args:
        model: Model name ('1b_lyrics', '5b', or '5b_lyrics')
        device: Device to load model on ('cuda' or 'cpu')
        hps: Hyperparameters dict
        levels: Which prior levels to load (None = all)
        auto_download: If True, automatically download missing checkpoints
    
    Returns:
        tuple: (vqvae, priors) model components
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODELS.keys())}")
    
    vqvae, *priors = MODELS[model]
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length=hps.get('sample_length', 0), sample_length_in_seconds=hps.get('sample_length_in_seconds', 0))), device, auto_download=auto_download)
    hps.sample_length = vqvae.sample_length
    if levels is None:
        levels = range(len(priors))
    # Load priors on the specified device (not hardcoded to 'cpu')
    priors = [make_prior(setup_hparams(priors[level], dict()), vqvae, device, auto_download=auto_download) for level in levels]
    return vqvae, priors
