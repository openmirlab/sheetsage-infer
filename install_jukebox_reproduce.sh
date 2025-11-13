#!/bin/bash
# Install jukebox with newer dependencies (bypasses numba==0.48.0 requirement)

set -e

pip install numba librosa
pip install git+https://github.com/chrisdonahue/jukebox.git@7e0a38b679ff3f64987d8297d9d0eb5a046880c1 --no-deps
pip install fire tqdm soundfile unidecode mpi4py

python3 -c "import jukebox, numba; print(f'✓ Jukebox installed (numba {numba.__version__})')"
