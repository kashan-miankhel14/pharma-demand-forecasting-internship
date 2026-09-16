# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately through **GitHub Security > Report a vulnerability**. Include the affected version, location, reproduction steps, impact, and any suggested mitigation. Do not publish sensitive details in public issues; we will coordinate validation and fixes through the private report.

## Secret handling

Never commit credentials, tokens, API keys, private keys, or connection strings. Supply required values through environment variables or an ignored local `.env` file. If a secret is exposed, revoke or rotate it immediately and remove it from repository history.

## Model and artifact trust

Treat serialized model files (including `.joblib`) as executable. Train and load artifacts only from trusted sources, verify their provenance and integrity, and never deserialize artifacts from untrusted repositories or uploads.

## API exposure

The API has no built-in authentication, authorization, rate limiting, or CORS policy and is intended for trusted local use. It binds to localhost and disables interactive documentation by default. Do not expose it publicly without authentication, TLS, input limits, rate limiting, and appropriate network controls.

## Data privacy

Use synthetic, de-identified, or otherwise authorized data and retain only what is necessary. Do not commit or log patient, prescriber, supplier, transaction-level, or other sensitive records. Apply access controls and defined retention practices to any real data or derived artifacts.

## Supported versions

Security fixes are supported for the current `main` branch and the latest published release, if any. Older versions and forks are not guaranteed to receive fixes; upgrade to a supported version promptly.

This repository is an internship demonstration, not a production inventory or clinical decision system.
