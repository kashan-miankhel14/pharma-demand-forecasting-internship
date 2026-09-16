# Data policy

The checked-in `sample_pharmacy_sales.csv` is a deterministic, tiny synthetic data set created for local development, tests, and smoke checks. It contains fictional medicine and branch codes and does not represent a real pharmacy, patient, supplier, or transaction.

The source project's raw CSV files are intentionally not copied into this repository. Their provenance and commercial sensitivity could not be verified from the files alone, so they are excluded rather than published. Add private data under `data/raw/` only when the data owner has approved its use and the data has been reviewed for personal, commercial, or regulated information. `data/raw/` is ignored by Git.

The schema expected by the package is documented in `docs/data.md`.
