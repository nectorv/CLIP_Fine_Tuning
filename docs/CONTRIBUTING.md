# Contributing to CLIP Fine-Tuning

Thank you for your interest in contributing! This document provides guidelines for participating in the project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Assume good intentions
- Report violations to maintainers

## Getting Started

### Fork & Setup

```bash
# Fork the repository on GitHub

# Clone your fork
git clone https://github.com/YOUR_USERNAME/CLIP_Fine_Tuning.git
cd CLIP_Fine_Tuning

# Add upstream remote
git remote add upstream https://github.com/nectorv/CLIP_Fine_Tuning.git

# Create feature branch
git checkout -b feature/your-feature-name
```

### Development Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install black isort pytest pytest-cov

# Setup pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

## Development Workflow

### 1. Code Style

Follow PEP 8 with these tools:

```bash
# Format code
black src/

# Sort imports
isort src/

# Check code quality
flake8 src/

# Type checking (if using type hints)
mypy src/
```

**Style Guidelines**:
- Max line length: 88 characters (Black default)
- Use type hints for function signatures
- Document complex logic with docstrings
- Remove debug code before committing

### 2. Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `perf`: Performance improvements
- `chore`: Build/dependency changes

**Example**:
```
feat(training): add gradient accumulation support

- Implement gradient accumulation for large batch sizes
- Add --grad_accum_steps argument to CLI
- Update documentation with usage examples

Closes #42
```

### 3. Testing

Write tests for new features:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cleaner.py -v

# Run with coverage
pytest tests/ --cov=src/ --cov-report=html

# Run test on changes
pytest-watch tests/
```

**Test Structure**:
```python
# tests/test_module.py
import pytest
from src.module import function_to_test

class TestFeature:
    def setup_method(self):
        """Setup before each test"""
        self.test_data = {...}
    
    def test_basic_functionality(self):
        """Test happy path"""
        result = function_to_test(self.test_data)
        assert result == expected
    
    def test_error_handling(self):
        """Test error cases"""
        with pytest.raises(ValueError):
            function_to_test(invalid_data)
```

### 4. Documentation

Update documentation for new features:

- **API Documentation**: [docs/API.md](API.md)
- **Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **Quick Start**: [docs/QUICKSTART.md](QUICKSTART.md)
- **README**: Update [README.md](../README.md)
- **Docstrings**: Add to all public functions

**Docstring Format** (Google style):
```python
def function_name(param1: str, param2: int) -> bool:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When validation fails
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
    pass
```

## Contributing Areas

### Bug Fixes
1. Open an issue describing the bug
2. Create a branch: `git checkout -b fix/bug-description`
3. Include a test that reproduces the bug
4. Fix the issue
5. Submit PR with reference to issue

### New Features
1. Open an issue to discuss the feature
2. Wait for feedback from maintainers
3. Create a branch: `git checkout -b feat/feature-name`
4. Implement with tests and documentation
5. Submit PR with detailed description

### Documentation
- Fix typos
- Add examples
- Clarify complex sections
- Update outdated information

**No review needed for**:
- Typo fixes
- README improvements
- Documentation clarifications

### Optimization
- Performance improvements
- Memory efficiency
- Training speed
- Inference latency

**Benchmarking Required**:
- Before/after metrics
- Testing environment details
- Hardware specifications

## Pull Request Process

### Before Submitting

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Format code**:
   ```bash
   black src/
   isort src/
   ```

4. **Check documentation**:
   ```bash
   # Build docs locally (if applicable)
   cd docs && make html
   ```

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Performance improvement

## Related Issue
Closes #123

## Testing
- [ ] Added/updated tests
- [ ] Tests pass locally
- [ ] Tested on different inputs

## Documentation
- [ ] Updated README
- [ ] Updated API documentation
- [ ] Added/updated docstrings
- [ ] Added code comments

## Checklist
- [ ] Code follows style guidelines
- [ ] No new warnings generated
- [ ] Tests added for new features
- [ ] Documentation is updated
- [ ] Branch is up to date with main
```

### PR Review

- Respond to feedback constructively
- Push fixes as new commits (don't force push)
- Request re-review when addressing comments
- Be patient - reviews take time

## Issue Guidelines

### Bug Reports

Include:
- Python version & OS
- Exact error message
- Minimal reproduction code
- Expected vs actual behavior
- Environment details (GPU, CUDA version, etc.)

**Example**:
```markdown
## Bug Description
Training fails with CUDA out of memory error

## Environment
- Python 3.10
- PyTorch 2.0.1
- CUDA 12.0
- GPU: RTX 3090

## Reproduction
```python
python -m src.training.main --scenario dual_lora --batch_size 256
```

## Error
```
RuntimeError: CUDA out of memory...
```

## Attempted Fixes
- Reduced batch_size to 128 - still fails
- set CUDA_LAUNCH_BLOCKING=1 - no change
```

### Feature Requests

Include:
- Problem statement
- Proposed solution
- Use case / motivation
- Potential implementation approach

**Example**:
```markdown
## Problem
Currently only supports CLIP ViT-B/32. Other architectures would be beneficial.

## Proposal
Add support for CLIP ViT-L/14 and ResNet-50 models

## Motivation
Enables accuracy/efficiency tradeoffs for different applications

## Implementation
- Modify ModelConfig to accept model_name parameter
- Update model loading in model.py
- Test with different model sizes
```

## Release Process

Maintainers handle releases, but contributors should:
- Follow semantic versioning
- Update CHANGELOG.md
- Document breaking changes
- Tag releases appropriately

## Resources

- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Git Documentation](https://git-scm.com/doc)

## Getting Help

- **Questions**: Open a Discussion or GitHub Issue (tag as question)
- **Technical Help**: Check existing issues/documentation
- **Contact Maintainers**: Email or DM on GitHub

## Acknowledgments

Contributors will be recognized in:
- CONTRIBUTORS.md
- Release notes
- GitHub contributors page

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

**Happy contributing!** 🎉
