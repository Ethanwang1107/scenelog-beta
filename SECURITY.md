# Security Policy

## Supported Versions

Scenelog is currently in public beta. Security fixes target the latest beta
release only.

| Version | Supported |
| --- | --- |
| 0.10.x | Yes |
| older versions | No |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Report security issues privately by contacting the maintainer through GitHub.
Include:

- Scenelog version
- macOS version and Mac model
- Clear reproduction steps
- Impact and affected local files or services

Do not include private videos, photos, transcripts, voice samples, credentials,
API keys, or other sensitive material in the report.

## Local Data Boundary

Scenelog is designed as a local-first tool:

- The local web server listens on `127.0.0.1` by default.
- Source media, people photos, voice samples, transcripts, indexes, and Excel
  outputs are stored on the user's Mac.
- Users should review third-party local model tools such as Ollama and
  whisper.cpp according to their own security requirements.

