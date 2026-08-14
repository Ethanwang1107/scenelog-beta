# Contributing to Scenelog

Scenelog is in public beta. Bug reports, installation notes, and focused pull
requests are welcome.

## Before Opening an Issue

Please check:

- You are using an Apple Silicon Mac with macOS 13 or later.
- Required local tools and models are installed.
- The Scenelog workspace shows the environment diagnostic result.
- The issue does not contain private media, photos, transcripts, voice samples,
  credentials, or personal data.

Useful issue details:

- Scenelog version
- macOS version and Mac model
- Whether you used the DMG or source install
- Last 20 lines of the process log
- Minimal reproduction steps

## Development Setup

```zsh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
```

## Pull Requests

Please keep pull requests focused. Include tests when changing pipeline,
indexing, state migration, Excel merging, people recognition, speaker
identification, or web API behavior.

Do not commit:

- Local media
- `_scenelog/` outputs
- Model files
- `.venv/`
- Build artifacts
- Private user data

