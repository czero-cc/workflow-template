# Contributing to CZero Engine Python SDK

Thank you for your interest in contributing to the CZero Engine Python SDK! We welcome contributions from the community.

## 🚀 Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/czero/workflow-template.git
   cd workflow-template
   ```

2. **Set Up Development Environment**
   ```bash
   # Install UV package manager
   pip install uv
   
   # Install dependencies
   uv pip install -e ".[dev]"
   ```

3. **Verify Setup**
   ```bash
   # Run tests
   uv run pytest
   
   # Check code style
   uv run ruff check .
   ```

## 📝 Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Write clean, readable code
- Follow existing code patterns
- Add type hints for all functions
- Update documentation as needed

### 3. Test Your Changes
```bash
# Run all tests
uv run pytest

# Run specific test
uv run pytest tests/test_integration.py::test_your_feature

# Check coverage
uv run pytest --cov=czero_engine --cov-report=html
```

### 4. Submit Pull Request
- Push your branch to your fork
- Create a pull request with clear description
- Link any related issues

## 🎯 Guidelines

### Code Style
- Use Python 3.11+ features
- Follow PEP 8 conventions
- Maximum line length: 100 characters
- Use descriptive variable names

### Testing
- Write tests for new features
- Maintain or improve code coverage
- Test edge cases and error handling
- Use async/await consistently

### Documentation
- Update docstrings for new functions
- Add examples for complex features
- Keep README.md current
- Document breaking changes

## 🏗️ Project Structure

```
workflow-template/
├── czero_engine/       # Main SDK package
│   ├── client.py       # API client
│   ├── models.py       # Pydantic models
│   └── workflows/      # High-level workflows
├── tests/              # Test suite
├── examples/           # Usage examples
└── docs/              # Additional documentation
```

## 🧪 Testing Requirements

All contributions must:
- Pass existing tests
- Include tests for new features
- Maintain 80%+ code coverage
- Handle errors gracefully

## 📦 Submitting Changes

### Pull Request Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated (see below)
- [ ] Commit messages are clear
- [ ] PR description explains changes

#### What "Documentation is updated" means:
Update documentation when your changes affect:
- **Docstrings**: Add/update function and class docstrings in your code
- **README.md**: Update if you add new features, change SDK usage, or improve examples
- **Examples**: Update or add example scripts if you introduce new functionality
- **Type hints**: Ensure all new functions have proper type annotations
- **CHANGELOG.md**: Add entry for breaking changes or major features (if file exists)

Examples:
- Adding a new workflow? → Update README.md with usage example
- New client method? → Add docstring with parameters and return type
- Improved error handling? → Update relevant documentation
- Fixed a common issue? → Consider adding to troubleshooting section

Note: The CZero Engine API is closed source and cannot be modified by external contributors. This SDK is a client library that interfaces with the existing API.

### Commit Message Format
```
type: brief description

Longer explanation if needed
Fixes #issue_number
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

## 🔧 Development Tips

### Running CZero Engine Locally
1. Download CZero Engine from [czero.cc](https://czero.cc)
2. Start the application
3. Ensure API server is running on port 1421
4. Load required models through the UI

### Debug Mode
```python
# Enable verbose logging
client = CZeroEngineClient(verbose=True)

# Use environment variables
CZERO_API_URL=http://localhost:1421
CZERO_VERBOSE=true
```

### Common Issues
- **Connection refused**: Ensure CZero Engine is running
- **Model not loaded**: Load models through the app UI
- **Timeout errors**: Increase client timeout for LLM operations

## 🤝 Code of Conduct

### Our Standards
- Be respectful and inclusive
- Welcome newcomers
- Accept constructive criticism
- Focus on what's best for the community

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or insulting comments
- Public or private harassment
- Publishing private information

## 📋 Issue Reporting

When reporting issues, include:
1. Python version and OS
2. CZero Engine version
3. Steps to reproduce
4. Error messages/logs
5. Expected vs actual behavior

## 🎖️ Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Project documentation

## 📞 Getting Help

- 💬 [Discord Community](https://discord.gg/yjEUkUTEak)
- 🐛 [Issue Tracker](https://github.com/czero/workflow-template/issues)
- 📧 [Email](mailto:info@czero.cc)

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to CZero Engine! 🚀