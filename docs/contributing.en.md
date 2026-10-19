[中文](contributing.zh.md)

# Add a model or experiment

1. Register weight repository, full revision, precision and context in `models/<model-id>/model.json`; provide `README.zh.md` and `README.en.md`.
2. Create `experiments/<model-id>/<date-engine-precision-topology>/` for each run. Copy `templates/experiment.json`; fill hardware, engine, configuration and objective. Never reuse stale dates, paths or model identity.
3. Freeze workload, seed, repeats, arrivals, deadlines and stops before execution. Distinguish fixed sample counts from conditional request caps.
4. Keep attempts separate from valid analysis cohorts. Retain failures, rejections, unexecuted/unsupported cases and missing fields. Retry success never rewrites first-attempt success.
5. Add bilingual conclusions, README, protocol and reports. Link every claim to data with workload, sample count, timing boundary, unmeasured conditions and future work.
6. Generate artifact SHA-256 in `artifacts.json`, update the model index, run `python scripts/verify_repository.py` and offline tests. PRs should explain condition changes and validation scope.

Compute speed ratios only when model revision, precision, lengths, concurrency, cache, topology and timing boundaries are compatible. Separate nominal specifications from measurements. Unmatched MTP/platform observations do not establish causal gains.

## GitHub automation

The [workflow template](../deploy/github-actions.validate.yml) runs offline tests, evidence checks and dry-runs. It is not enabled in GitHub Actions because the publishing credential lacks the `workflow` scope. A repository maintainer with workflow-write permission can save it as `.github/workflows/validate.yml`. It does not call model endpoints or execute GPU workloads. See the [local publication validation receipt](../provenance/publication-validation.json).

## Publication privacy checks

Run `python scripts/verify_repository.py` to check bare domains, URLs, non-example private addresses, personal paths, SSH usernames and embedded SVG. Use `example.invalid`, documentation addresses or environment variables for endpoints, and placeholders such as `benchmark` for users. Documentary domains require explicit review in the allowlist. Never put real credentials or deployment identities in test fixtures.
