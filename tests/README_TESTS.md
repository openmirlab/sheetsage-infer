# SheetSage Test Suite

Comprehensive pytest test suite for the SheetSage inference package.

## Test Files Overview

### 1. **test_imports.py** - Import Tests
Tests that all modules can be imported correctly.

```bash
pytest tests/test_imports.py -v
```

**What it tests:**
- Core module imports (`sheetsage`, `Status`)
- Representation imports (`Handcrafted`, `JukeboxEmbeddings`)
- Theory class imports (`LeadSheet`, `Melody`, `Harmony`, etc.)
- Utility function imports (`engrave`, `retrieve_audio_bytes`)
- Neural module imports (`EncOnlyTransducer`, etc.)

### 2. **test_signatures.py** - API Signature Tests
Tests that functions have correct parameters and signatures.

```bash
pytest tests/test_signatures.py -v
```

**What it tests:**
- `sheetsage()` function parameters match notebook usage
- Feature extractor callable interfaces
- Theory class method signatures (`as_lily()`, `as_midi()`)
- Utility function signatures

### 3. **test_representations.py** - Feature Extractor Tests
Tests feature extraction modules.

```bash
pytest tests/test_representations.py -v
```

**What it tests:**
- Handcrafted extractor initialization
- JukeboxEmbeddings class structure
- Inheritance from base `Representation` class
- InputFeats enum values

### 4. **test_integration_full.py** - Full Integration Tests
Tests complete inference pipeline with real audio (slow).

```bash
# Run integration tests (requires --run-slow flag)
pytest tests/test_integration_full.py --run-slow -v

# With Jukebox (requires GPU)
pytest tests/test_integration_full.py --run-slow --run-jukebox -v
```

**What it tests:**
- Full inference with Handcrafted features
- Full inference with Jukebox features (GPU required)
- Notebook usage patterns (Cell 0, 1, 2)
- Output types and lengths
- Edge cases and error handling

### 5. **test_output_formats.py** - Output Format Tests
Tests output conversions (LilyPond, MIDI, etc.).

```bash
pytest tests/test_output_formats.py -v

# With LilyPond binary installed
pytest tests/test_output_formats.py --with-lilypond -v
```

**What it tests:**
- LilyPond export (`as_lily()`)
- MIDI export (`as_midi()`)
- Engraving to PNG/PDF (requires LilyPond)
- TheoryTab format conversions
- Saving outputs to files

### 6. **test_theory_classes.py** - Music Theory Tests
Tests music theory classes.

```bash
pytest tests/test_theory_classes.py -v
```

**What it tests:**
- Note, Chord creation
- Melody, Harmony classes
- Key, Meter, Tempo classes
- LeadSheet creation and methods

### 7. **conftest.py** - Pytest Configuration
Pytest configuration with fixtures and custom options.

**Provides:**
- Custom command-line flags
- Test markers (slow, gpu, jukebox, lilypond)
- Shared fixtures (test_audio_url, test_params, etc.)

## Running Tests

### Quick Start

```bash
# Run fast tests only (imports, signatures, theory classes)
pytest tests/ -v

# Run all tests including slow integration tests
pytest tests/ --run-slow -v

# Run with coverage
pytest tests/ --cov=sheetsage --cov-report=html
```

### Command-Line Options

```bash
# Run slow integration tests
pytest --run-slow

# Run Jukebox/GPU tests
pytest --run-jukebox

# Run tests requiring LilyPond
pytest --with-lilypond

# Combine options
pytest --run-slow --run-jukebox --with-lilypond
```

### Using Markers

```bash
# Run only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run only GPU tests
pytest -m gpu

# Run tests NOT requiring GPU
pytest -m "not gpu"
```

### Specific Test Selection

```bash
# Run specific file
pytest tests/test_imports.py

# Run specific class
pytest tests/test_imports.py::TestCoreImports

# Run specific test
pytest tests/test_imports.py::TestCoreImports::test_import_sheetsage_function

# Run tests matching pattern
pytest tests/ -k "import"
pytest tests/ -k "signature"
pytest tests/ -k "jukebox"
```

## Test Organization

### Fast Tests (< 1 second)
- `test_imports.py` - All import tests
- `test_signatures.py` - All signature tests
- `test_theory_classes.py` - Theory class tests
- `test_representations.py` - Basic structure tests

### Slow Tests (requires `--run-slow`)
- `test_integration_full.py` - Full inference pipeline tests
- Download and process real audio files
- May take 30+ seconds per test

### GPU Tests (requires `--run-jukebox`)
- Jukebox feature extraction tests
- Requires CUDA-capable GPU with >=12GB VRAM
- Jukebox dependencies installed

### External Dependency Tests
- `--with-lilypond` - Tests requiring LilyPond binary

## Continuous Integration Setup

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov

    - name: Run fast tests
      run: pytest tests/ -v --cov=sheetsage

    - name: Run slow tests
      run: pytest tests/ --run-slow -v
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

## Writing New Tests

### Template for New Test File

```python
#!/usr/bin/env pytest
"""Description of test module."""

import pytest


class TestFeatureName:
    """Test specific feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        from sheetsage.module import function

        # Act
        result = function()

        # Assert
        assert result is not None

    @pytest.mark.slow
    def test_integration(self, test_audio_url):
        """Test integration (slow)."""
        # Uses fixtures from conftest.py
        pass

    @pytest.mark.gpu
    def test_gpu_feature(self):
        """Test GPU-dependent feature."""
        try:
            # Test code
            pass
        except RuntimeError as e:
            if "CUDA" in str(e):
                pytest.skip("GPU not available")
            raise
```

## Test Fixtures

Available fixtures from `conftest.py`:

- `test_audio_url` - Test audio URL
- `test_params` - Common test parameters
- `handcrafted_params` - Parameters for handcrafted tests
- `jukebox_params` - Parameters for Jukebox tests
- `tmp_path` - Temporary directory (pytest built-in)

## Test Markers

- `@pytest.mark.slow` - Slow tests (requires --run-slow)
- `@pytest.mark.gpu` - GPU tests (requires --run-jukebox)
- `@pytest.mark.jukebox` - Jukebox tests
- `@pytest.mark.lilypond` - LilyPond tests (requires --with-lilypond)
- `@pytest.mark.skip()` - Skip test unconditionally
- `@pytest.mark.skipif()` - Skip test conditionally

## Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=sheetsage --cov-report=html

# View in browser
open htmlcov/index.html

# Generate terminal report
pytest tests/ --cov=sheetsage --cov-report=term-missing

# Generate XML for CI tools
pytest tests/ --cov=sheetsage --cov-report=xml
```

## Debugging Tests

```bash
# Stop at first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Enter debugger on failure
pytest tests/ --pdb

# Verbose output
pytest tests/ -vv

# Show print statements
pytest tests/ -s
```

## Best Practices

1. **Keep tests independent** - Each test should run in isolation
2. **Use fixtures** - Share setup code via fixtures
3. **Mark slow tests** - Use `@pytest.mark.slow` for integration tests
4. **Skip gracefully** - Handle missing dependencies with pytest.skip()
5. **Test edge cases** - Include error conditions and boundary cases
6. **Use descriptive names** - Test names should describe what they test
7. **Follow AAA pattern** - Arrange, Act, Assert

## Common Issues

### ImportError: No module named 'sheetsage'
```bash
# Install package in development mode
pip install -e .
```

### Jukebox tests fail with CUDA error
```bash
# Skip GPU tests if no GPU available
pytest tests/ -m "not gpu"
```

### Slow tests are skipped
```bash
# Enable slow tests explicitly
pytest tests/ --run-slow
```

### LilyPond tests fail
```bash
# Install LilyPond or skip those tests
pytest tests/ -m "not lilypond"
```

## Summary

Total test count by category:
- **Import tests**: ~20 tests
- **Signature tests**: ~10 tests
- **Representation tests**: ~10 tests
- **Integration tests**: ~10 tests
- **Output format tests**: ~10 tests
- **Theory class tests**: ~15 tests

**Total**: ~75 comprehensive tests covering all aspects of SheetSage!
