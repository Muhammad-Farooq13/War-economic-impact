# Contributing

Thank you for considering contributing to the War Economic Impact Predictor!

## How to Contribute

### 1. Fork & Clone

```bash
git clone https://github.com/Muhammad-Farooq13/War-economic-impact.git
cd war-economic-impact
```

### 2. Set Up Dev Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
make install-dev
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Follow existing code style (black, isort enforced via pre-commit)
- Add or update tests for any new functionality
- Update `config/config.yaml` if adding new settings
- Keep notebooks clean (clear outputs before committing)

### 5. Run Quality Checks

```bash
make format    # auto-format code
make lint      # flake8 check
make test      # run pytest
```

All checks must pass before submitting a PR.

### 6. Submit a Pull Request

- Write a clear PR title and description
- Reference any related issues
- Include before/after metrics if changing model parameters

## Code Style

- **Formatter:** `black` (line length 100)
- **Import sorter:** `isort --profile black`
- **Linter:** `flake8` (max-line-length 100, ignore E203,W503)
- **Type hints:** required for all public functions/methods
- **Docstrings:** Google-style for public APIs

## Reporting Bugs

Please open a GitHub Issue with:
- Python version (`python --version`)
- OS
- Steps to reproduce
- Expected vs actual behaviour
- Relevant stack trace
