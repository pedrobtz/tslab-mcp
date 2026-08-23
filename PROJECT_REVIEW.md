# Project Review: `tslab-mcp`

Reviewed: 2026-08-22

## Assessment

This is a well-designed, unusually disciplined alpha MVP. The product direction
is coherent, the architecture supports it, and verification is strong. The
issues found in the original review have now been remediated in the working
tree; publication and a tagged release remain separate maintainer actions.

## Remediation status

- MCP dependency floor raised to the v2 API actually used by the server;
  StatsForecast's floor was likewise aligned with the verified 2.1.1 backend.
- Canonical aliases enforced, including `AutoCES`.
- Default conformal intervals added for ADIDA, IMAPA, and CrostonClassic.
- Tool annotations corrected for mutation, destructiveness, and network reach.
- Strict required-value/frequency checks and timestamp-gap diagnostics added.
- Irregular panels blocked from analysis without automatic imputation.
- Model/window history preflights and finite-output validation added.
- An all-model statistical smoke matrix now covers forecast intervals,
  cross-validation names/metrics, and anomaly flags.
- A packaged MCP instruction contract now directs external agents through the
  TimeCopilot-style feature, evaluation, selection, forecast, and anomaly loop,
  followed by reproducible export.

## Goal

`tslab-mcp` turns deterministic time-series forecasting into nine MCP tools:

```text
load -> describe -> discover models -> cross-validate -> validate selection
     -> forecast/anomalies -> manifest/report
```

The agent remains responsible for reasoning and model choice; Python libraries
produce the numbers. Large frames stay behind handles or in Parquet artifacts,
while manifests preserve input hashes, parameters, versions, and the
model-selection rationale. This is clearly articulated in [README.md](README.md).

The design is strong:

- Lightweight StatsForecast core with optional TimeCopilot/foundation models.
- Lazy heavyweight imports.
- Blocking forecasting work moved off the event loop.
- Bounded MCP responses.
- Auditable manifests and offline reports.
- Actionable errors and strict input schemas.

## Current status

- Version: `0.1.0`, classified as alpha in [pyproject.toml](pyproject.toml).
- Nine tools are implemented end-to-end.
- The remediation is implemented in the working tree and has not been committed
  or pushed.
- Local verification:
  - 235 base tests passed; 6 optional TimeCopilot tests were deselected on the
    current Intel Mac, where upstream PyTorch has no compatible wheel.
  - Ruff passed.
  - Strict mypy passed.
  - Wheel and source distribution built successfully.
  - The installed wheel was exercised at its MCP 2.0.0 and StatsForecast 2.1.1
    dependency floors.
  - A real stdio MCP client connected, discovered all nine tools, and
    successfully called `tsf_list_models`.
- The last remote code-bearing [GitHub CI run succeeded][ci-run], covering
  Python 3.10-3.13 and the optional foundation integration. The current
  working-tree changes still need that remote matrix after they are committed.
- The repository is still very young: four commits, no open issues or pull
  requests, no tags, and [no GitHub release][releases]. The documentation now
  leads with Git installation and labels the shorter PyPI command as available
  only after the first release.

## Important findings

### 1. Resolved: the MCP dependency floor was incorrect

The review found that [pyproject.toml](pyproject.toml) permitted `mcp>=1.2.0`
while [server.py](src/tslab_mcp/server.py) used the v2-only `MCPServer` API. An
import failure was reproduced with MCP 1.29.0. The requirement is now
`mcp>=2.0.0`; MCP v2 is a breaking rewrite of v1. See the [official MCP package
information][mcp-pypi].

### 2. Resolved: `AutoCES` could not be evaluated correctly

StatsForecast defaults the model's prediction column to `CES`, while the public
registry name is `AutoCES`. The backend now passes every public model name as
its canonical alias, and output validation rejects future alias drift before an
artifact is recorded.

### 3. Resolved: three models failed with the default forecast request

`ForecastInput` defaults to 80% and 95% intervals in
[schemas.py](src/tslab_mcp/schemas.py). `ADIDA`, `IMAPA`, and `CrostonClassic`
have no native interval implementation, so [backends.py](src/tslab_mcp/backends.py)
now supplies two deterministic conformal-distribution calibration windows and
records that configuration in the manifest.

### 4. Resolved: MCP safety annotations did not match actual behavior

[server.py](src/tslab_mcp/server.py) now marks only model listing read-only.
Stateful and artifact-producing tools are non-idempotent; handle replacement is
potentially destructive; and CV, forecasting, and anomaly detection are
conservatively open-world because they can reach model downloads or TimeGPT.

MCP defines read-only as not modifying the environment. See the [official
`ToolAnnotations` specification][tool-annotations].

### 5. Resolved: input and output validation was incomplete

[registry.py](src/tslab_mcp/registry.py) now:

- Rejects missing/blank IDs, invalid timestamps, and nonnumeric/non-finite
  targets.
- Validates explicit pandas frequency aliases.
- Reports missing and off-grid timestamps per panel and blocks analysis until
  irregular source data is explicitly regularized.
- Preflights statistical model/window history, including verified per-model
  minima, and validates structural, canonical, finite backend points and
  intervals plus boolean anomaly flags before writing artifacts or successful
  run records.

## Design decisions made

The mechanical defects had direct fixes: correct dependency floors, canonical
aliases, truthful annotations, stricter required-value validation, and CI
assertions. Four findings required policy choices, now made explicit:

- Irregular data is diagnostic-only. Loading reports the exact gaps, but
  analytical tools stop until the caller regularizes the source; the server
  never chooses filling or imputation on the user's behalf.
- Intermittent models keep the default interval contract through deterministic
  conformal-distribution intervals with two calibration windows, and the
  manifest records that choice.
- Default anomaly coverage is the largest window count that preserves verified
  model-specific, seasonal, and interval-calibration history rather than the
  mathematically largest window count.
- Model selection is a deterministic checkpoint after CV: the agent proposes a
  model, Python verifies the lowest finite score and required improvement over
  SeasonalNaive, and unresolved or rejected decisions block forecasting unless
  the caller marks an explicit user-directed override.

## Recommended next milestone

The hardening work identified by this review is complete. Before publishing:

1. Run the complete CI matrix, including the optional TimeCopilot integration.
2. Review release metadata and choose the final `0.1.0` commit.
3. Tag and publish the package using trusted publishing.
4. Confirm the documented short `uvx tslab-mcp` path after publication.

With the remediations above, the codebase is a credible first-release candidate.

[ci-run]: https://github.com/pedrobtz/tslab-mcp/actions/runs/32181415019
[releases]: https://github.com/pedrobtz/tslab-mcp/releases
[mcp-pypi]: https://pypi.org/project/mcp/
[tool-annotations]: https://modelcontextprotocol.io/specification/2025-11-25/schema
