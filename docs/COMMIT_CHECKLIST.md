# Pre-Commit Checklist

**Date:** 2026-01-01

## Files to Include

### ✅ Core Configuration Files
- [x] `uv.lock` - Dependency lock file for reproducible builds
- [x] `pyproject.toml` - Project configuration and dependencies
- [x] `.gitignore` - Git ignore patterns
- [x] `LICENSE` - MIT License file
- [x] `README.md` - Project documentation

### ✅ Package Source Code
- [x] `sheetsage/` - Entire package directory including:
  - `__init__.py`
  - `infer.py` - Main inference function
  - `assets.py` - Asset management
  - `beat_track.py` - Beat tracking
  - `utils.py` - Utility functions
  - `align.py` - Beat alignment
  - `modules/` - Neural network modules
  - `representations/` - Feature extractors (handcrafted + jukebox)
  - `theory/` - Music theory classes
  - `assets/` - Asset JSON files

### ✅ Examples (Excluding Test Scripts)
- [x] `examples/__init__.py`
- [x] `examples/basic_transcription.py` - Basic usage example
- [x] `examples/hooktheory_example.py` - Hooktheory data example
- [x] `examples/hooktheory_simple.py` - Simple Hooktheory example
- [x] `examples/jukebox_transcription.py` - Jukebox usage example

### ❌ Excluded from Examples
- [ ] `examples/test_jukebox_cuda.py` - Test script (exclude)
- [ ] `examples/test_jukebox_cuda_diagnostic.py` - Diagnostic test (exclude)

## Potentially Missing Files

### ⚠️ Optional but Recommended
- [ ] `requirements.txt` - Alternative dependency file (if you want pip compatibility)
  - Note: `pyproject.toml` already has dependencies, but some users prefer `requirements.txt`
  - Can be auto-generated from `pyproject.toml` if needed

### ⚠️ Documentation (Optional)
- [ ] `CHANGELOG.md` - Version history (if you maintain one)
- [ ] `CONTRIBUTING.md` - Contribution guidelines (if accepting contributions)
- [ ] `docs/` - Additional documentation (currently in `docs/generated/`)

### ⚠️ CI/CD (Optional)
- [ ] `.github/workflows/` - GitHub Actions workflows (if using CI)
- [ ] `.pre-commit-config.yaml` - Pre-commit hooks (if using pre-commit)

### ⚠️ Development Tools (Optional)
- [ ] `test_before_push_py310.sh` - Pre-push test script (useful for contributors)
- [ ] `docs/TEST_BEFORE_PUSH.md` - Testing documentation (useful for contributors)

## Files to Exclude (Already in .gitignore)

These should NOT be committed:
- `__pycache__/` directories
- `.venv/`, `venv/`, `.venv_py310/` - Virtual environments
- `dist/`, `build/`, `*.egg-info/` - Build artifacts
- `*.wav`, `*.mp3` - Audio files
- `*.pth`, `*.pt` - Model files
- `hooktheory_transcription_results/` - Generated outputs
- `jukebox_test_output/` - Test outputs
- `.pytest_cache/` - Test cache

## Verification Commands

Before committing, verify:

```bash
# 1. Check what will be committed
git status

# 2. Verify package structure
python -c "import sheetsage; print('✓ Package imports work')"

# 3. Verify examples (excluding tests)
ls examples/*.py | grep -v test_

# 4. Check for accidental inclusions
git status --ignored | grep -E "(venv|__pycache__|dist|build)"
```

## Summary

**Essential Files (Must Include):**
- ✅ `uv.lock`
- ✅ `pyproject.toml`
- ✅ `sheetsage/` (entire directory)
- ✅ `examples/` (excluding test scripts)
- ✅ `README.md`
- ✅ `LICENSE`
- ✅ `.gitignore`

**Total Essential Files:** ~7 items (plus entire directories)

**Optional Files (Consider Including):**
- `requirements.txt` (for pip compatibility)
- `test_before_push_py310.sh` (useful for contributors)
- `docs/TEST_BEFORE_PUSH.md` (documentation)


