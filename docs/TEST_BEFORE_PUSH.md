# Pre-Push Testing Guide

**Date:** 2025-12-21

This guide provides commands to test your code before pushing to git.

## Quick Start (Automated Script)

```bash
# Run the automated test script
./test_before_push.sh
```

## Manual Testing Steps

### Option 1: Using `uv` (Recommended - Faster)

```bash
# 1. Check git status
git status           # Should show clean working tree
git pull --rebase    # Make sure you're up to date

# 2. Create fresh virtual environment
rm -rf .venv
uv venv --python 3.12
source .venv/bin/activate

# 3. Install build tools and dev dependencies
uv pip install --upgrade pip build twine pytest pytest-cov ruff mypy

# 4. Install package in editable mode with dev dependencies
uv pip install -e ".[dev]"

# 5. Run linting
ruff check sheetsage/ tests/ examples/
ruff format --check sheetsage/ tests/ examples/

# 6. Run tests
pytest tests/ -v

# 7. Build package
rm -rf dist/ build/ *.egg-info/ sheetsage.egg-info/
python -m build

# 8. Verify package metadata
twine check dist/*

# 9. Test install from wheel
pip uninstall -y sheetsage 2>/dev/null || true
pip install dist/*.whl

# 10. Test import
python -c "import sheetsage; from sheetsage.infer import sheetsage; print('✓ Import successful')"
```

### Option 2: Using Standard `pip` and `venv`

```bash
# 1. Check git status
git status           # Should show clean working tree
git pull --rebase    # Make sure you're up to date

# 2. Create fresh virtual environment
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install build tools and dev dependencies
pip install --upgrade pip build twine pytest pytest-cov ruff mypy

# 4. Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# 5. Run linting
ruff check sheetsage/ tests/ examples/
ruff format --check sheetsage/ tests/ examples/

# 6. Run tests
pytest tests/ -v

# 7. Build package
rm -rf dist/ build/ *.egg-info/ sheetsage.egg-info/
python -m build

# 8. Verify package metadata
twine check dist/*

# 9. Test install from wheel
pip uninstall -y sheetsage 2>/dev/null || true
pip install dist/*.whl

# 10. Test import
python -c "import sheetsage; from sheetsage.infer import sheetsage; print('✓ Import successful')"
```

## Key Differences from Generic Template

1. **Package name**: `sheetsage` (not `your_package`)
2. **Installation**: Use `pip install -e ".[dev]"` instead of `requirements.txt`
   - The `[dev]` extra includes pytest, ruff, mypy, etc.
3. **Python version**: Uses Python 3.12 (or 3.10+ as specified in pyproject.toml)
4. **Linting**: Includes `ruff` checks (formatting and linting)
5. **Test directory**: `tests/` (not automatically discovered, specify explicitly)

## What Gets Tested

- ✅ Git working tree is clean
- ✅ Up to date with remote
- ✅ Fresh virtual environment
- ✅ Package installs correctly
- ✅ Linting passes (ruff)
- ✅ Formatting is correct (ruff format)
- ✅ All tests pass (pytest)
- ✅ Package builds successfully (build)
- ✅ Package metadata is valid (twine check)
- ✅ Package installs from wheel
- ✅ Package imports correctly

## Troubleshooting

### If tests fail:
```bash
# Run tests with more verbose output
pytest tests/ -v -s

# Run specific test file
pytest tests/test_imports.py -v
```

### If linting fails:
```bash
# Auto-fix what can be fixed
ruff check --fix sheetsage/ tests/ examples/

# Auto-format code
ruff format sheetsage/ tests/ examples/
```

### If build fails:
```bash
# Check pyproject.toml syntax
python -c "import tomli; tomli.load(open('pyproject.toml', 'rb'))"
```

### If import fails after install:
```bash
# Check what was installed
pip show sheetsage
pip list | grep sheetsage
```

## Clean Up

After testing, you can clean up generated files:

```bash
rm -rf dist/ build/ *.egg-info/ sheetsage.egg-info/ .venv/
```

## Notes

- The script uses `uv` if available (faster), otherwise falls back to standard `pip`
- All dependencies are defined in `pyproject.toml`, not `requirements.txt`
- The `[dev]` extra includes development dependencies (pytest, ruff, mypy)
- Tests are in the `tests/` directory


