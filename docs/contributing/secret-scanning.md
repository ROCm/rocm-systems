# Secret Scanning in PerfXpert

**Phase 9 Task #52** — Pre-commit secret scanning to prevent API key leaks.

## Overview

This project includes automated secret scanning at commit time and CI. The system detects and blocks commits containing:

- API keys: Anthropic (`sk-ant-api*`), OpenAI (`sk-proj-*`, `sk-*`), AWS (`AKIA*`)
- Tokens: GitHub (`ghp_*`), npm (`npm_*`)
- Environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `PERFXPERT_LLM_*_KEY`
- Large binary files (>5MB) that may contain sensitive data

## Pre-Commit Hook

The `.pre-commit-config.yaml` includes a `secret-scan` hook that runs on every commit:

```bash
python3 tools/_secret_scanner.py
```

To install hooks locally:

```bash
pip install pre-commit
pre-commit install
```

To run manually before committing:

```bash
bash docs/lint.sh
python3 tools/_secret_scanner.py
```

## Handling Accidental Secrets

If a secret is committed to a branch:

### 1. Stop Immediately

Do NOT push the branch.

### 2. Identify What Was Exposed

Check the commit history:

```bash
git log --all --oneline | grep <secret>
```

### 3. Remove the Secret

Option A: Amend the most recent commit (if not yet pushed):

```bash
# Edit the file to remove the secret
git add <file>
git commit --amend --no-edit
```

Option B: Create a new commit that removes the secret:

```bash
# Remove the secret and commit
git rm <file-with-secret>
git commit -m "chore: remove accidental secret from <file>"
```

### 4. Rotate the Exposed Secret

**Critical:** If any API key was exposed:

1. Log into the service (Anthropic, OpenAI, AWS, etc.)
2. Revoke the exposed key immediately
3. Generate a new key
4. Update your `.env` file or CI secret store with the new key

### 5. Clean Git History (Advanced)

For secrets that were in older commits, use `git filter-branch` or `git filter-repo`:

```bash
# Remove a file from all history
git filter-repo --invert-paths --path <file>

# Remove a pattern from all files
git filter-repo --replace-text replacements.txt
```

See [GitHub's guide to removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

### 6. Force-Push (Only if Not Yet Pushed)

```bash
git push --force-with-lease origin <branch>
```

**Warning:** Only do this if the branch has not been merged to `main` or been pulled by others.

## Scanner Rules

The secret scanner (`tools/_secret_scanner.py`) checks for:

| Pattern | Reason |
|---------|--------|
| `sk-ant-api[0-9a-zA-Z_\-]{20,}` | Anthropic API key |
| `sk-proj-[A-Za-z0-9_\-]{20,}` | OpenAI project key |
| `sk-[A-Za-z0-9_\-]{40,}` | Generic secret key |
| `AKIA[0-9A-Z]{16}` | AWS access key |
| `ghp_[A-Za-z0-9_]{36,}` | GitHub personal access token |
| `npm_[A-Za-z0-9_]{36,}` | npm token |
| `PERFXPERT_LLM_\w+_KEY=` | PerfXpert LLM key env var |
| Binary `.db` / `.sqlite` files > 5MB | May contain sensitive data |

## Environment Variable Sanitization

The `perfxpert.tools._safety.build_safe_env()` function automatically strips environment variables with these suffixes when building subprocess environments:

- `_API_KEY`
- `_TOKEN`
- `_SECRET`

This prevents accidental leaks through subprocess calls or logs.

### Example

```python
import os
from perfxpert.tools._safety import build_safe_env

os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-...'
os.environ['MY_SECRET'] = 'secret'
os.environ['PATH'] = '/usr/bin'

safe_env = build_safe_env()
# safe_env['PATH'] is present
# safe_env['ANTHROPIC_API_KEY'] is NOT present
# safe_env['MY_SECRET'] is NOT present
```

## CI Integration

GitHub Actions workflow `.github/workflows/perfxpert-pre-commit.yml` runs all checks on pull requests:

- Secret scanner
- Docs lint
- Link checker
- Sample executor

All checks must pass before merge.

## Exceptions

Limited exceptions for:

- Documentation files explaining secrets (without actual values)
- `.pre-commit-config.yaml` (metadata only)
- `.github/workflows/` (no secrets hardcoded)

## Reporting Security Issues

If you discover a secret in the codebase:

1. **Do not commit or push it**
2. File a security issue (not public issue): aelwazir@amd.com
3. Provide details in a private channel
4. Do not include the secret in any communication

---

**See Also:**
- `.pre-commit-config.yaml` — hook configuration
- `tools/_secret_scanner.py` — scanner implementation
- `perfxpert/tools/_safety.py` — env sanitization
- `.github/workflows/perfxpert-pre-commit.yml` — CI integration
