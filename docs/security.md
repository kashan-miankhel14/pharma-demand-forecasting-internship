# Security and privacy notes

- The API binds to `127.0.0.1` by default. `0.0.0.0` and `::` are rejected unless `ALLOW_PUBLIC_BIND=true` is set deliberately.
- Interactive docs and OpenAPI exposure are disabled by default. Set `API_DOCS_ENABLED=true` only for a trusted local development environment.
- The API has no authentication, authorization, rate limiting, or CORS policy. It is a local internship demo, not a public service.
- `.env` is ignored by Git. Put secrets in the environment or a local `.env` file only when needed; this project does not require secrets for its default workflow.
- Runtime `.joblib` model files are executable when loaded. Generate them from trusted data and never load artifacts from an untrusted repository or upload directory.
- Raw commercial CSVs and trained model binaries are excluded from Git. The checked-in sample is synthetic and contains no real pharmacy or patient records.
- API responses expose aggregate forecasts and model metadata only. Do not add patient, prescriber, supplier, or transaction-level endpoints without a privacy review.
