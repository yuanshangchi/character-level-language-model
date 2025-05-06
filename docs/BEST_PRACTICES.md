## Required Extensions

Make sure you have these up-to-date extensions installed:

- Python (`ms-python.python`)
- Black Formatter (`ms-python.black-formatter`)
- Ruff (`charliermarsh.ruff`)
- isort (`ms-python.isort`)

## `.vscode/settings.json`

```json
{
    // Editor settings
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": "always",
        "source.organizeImports": "always"
    },
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "files.exclude": {
        "**/__pycache__": true
    },
    // Python settings
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll": "always",
            "source.organizeImports": "always"
        }
    },
    // Black formatter settings
    "black-formatter.args": [
        "--line-length",
        "88"
    ],
    // isort settings
    "isort.args": [
        "--profile",
        "black"
    ],
    // Pylint settings
    "ruff.enable": true,
    "ruff.lint.select": [
        "F401",
        "F841",
        "I001"
    ],
}
```
## `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.1.0
    hooks:
      - id: black
        args: ["--line-length=88"]

  - repo: https://github.com/pycqa/isort
    rev: 6.0.1
    hooks:
      - id: isort
        args: ["--profile", "black"]

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.11.8
    hooks:
      - id: ruff
        args: ["--fix", "--select=F401,F841,I001"]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
```
