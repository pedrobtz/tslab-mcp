# Publishing `tslab-mcp` to PyPI

Steps to get a release onto PyPI, plus the things about *this* package that make
a release go wrong. Work top to bottom; the pre-flight section is where the real
risk lives.

---

## 0. Release readiness

Both blockers this section used to carry are now resolved. Kept as a record of
what was checked, and of what to re-check when TimeCopilot moves.

### The version is single-sourced ✅

`pyproject.toml` declares `dynamic = ["version"]` and hatchling reads it from
`src/tslab_mcp/__init__.py`, which is the only place a release number is edited.
This matters because `manifest.py` reports both the module's `__version__` and
the installed distribution metadata in the same lineage record, so drift would
make an exported manifest self-contradictory.

Verified by bumping the module to `9.9.9`, rebuilding, and getting a
`tslab_mcp-9.9.9` wheel. To bump a release: edit that one line, commit, tag.

### The supported Python range is tested ✅

CI runs the suite on 3.10, 3.11, 3.12 and 3.13 on every push, so
`requires-python = ">=3.10"` is exercised rather than asserted. The base install
resolves to 63 packages on 3.11–3.13 and 67 on 3.10, and all four report the
same test count.

### What still needs judgement at release time

- **Cross-backend parity.** `tests/test_parity.py` asserts the statsforecast and
  TimeCopilot backends agree to 1e-9 on forecasts, intervals, MASE/sMAPE and the
  anomaly column. It runs on the CI leg that installs the `foundation` extra.
  Confirm that leg is green before tagging; a silent divergence would mean a
  manifest exported from a base install does not reproduce on an extra install.
- **TimeCopilot upgrades.** Its dependency pins are what constrain this
  package's supported Python and pandas range, and it publishes no per-model
  extras, which is why `foundation` is all-or-nothing. Re-check both on any
  upgrade.

---

## 1. Pre-flight checklist

```bash
uv sync --all-groups
uv run ruff check src tests
uv run mypy
uv run pytest                # fast suite
uv run pytest -m slow        # exercises TimeCopilot, ~3.5 min
```

Then confirm the artifacts are well-formed:

```bash
rm -rf dist
uv build
uvx twine check dist/*       # validates metadata + README rendering
```

`twine check` catches the classic silent failure: a README that renders on
GitHub but breaks PyPI's stricter Markdown parser, leaving a project page of raw
text.

Inspect what you are actually shipping — the wheel should contain the thirteen
modules plus `py.typed`, and nothing else:

```bash
python -c "import zipfile;print('\n'.join(sorted(zipfile.ZipFile('dist/tslab_mcp-0.1.0-py3-none-any.whl').namelist())))"
```

---

## 2. Accounts and credentials

1. Register at <https://pypi.org/account/register/> and at
   <https://test.pypi.org/account/register/> (separate accounts, separate
   passwords).
2. Enable 2FA on both. PyPI requires it for uploads.
3. Choose a credential method:

**Trusted Publishing (recommended).** No long-lived token exists to leak. On
PyPI go to *Your projects → Publishing* and add a pending publisher:

| Field | Value |
|---|---|
| PyPI project name | `tslab-mcp` |
| Owner | `pedrobtz` |
| Repository | `tslab-mcp` |
| Workflow | `release.yml` |
| Environment | `pypi` |

**API token (simpler for a one-off).** *Account settings → API tokens*. Scope it
to the project once the project exists; the first upload needs an
account-scoped token because the project does not exist yet. Store it as
`UV_PUBLISH_TOKEN` rather than pasting it into a shell command, where it lands
in your history.

---

## 3. Rehearse on TestPyPI

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"
```

Then check the project page renders, the metadata reads correctly, and the
description is not raw Markdown.

**Do not expect a TestPyPI install to work.** TestPyPI does not mirror real
dependencies, and even the light base install needs statsforecast, utilsforecast
and mcp from real PyPI. A plain
`pip install -i https://test.pypi.org/simple tslab-mcp` will fail to resolve. To
test the install path, pull the package from TestPyPI and its dependencies from
real PyPI:

```bash
uv pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  --index-strategy unsafe-best-match \
  tslab-mcp
```

TestPyPI proves the *upload and metadata* work. It does not prove the package
installs cleanly; step 5 does that.

---

## 4. Publish

Tag the release so the artifact is traceable to a commit:

```bash
git tag -a v0.1.0 -m "tslab-mcp 0.1.0"
git push origin v0.1.0
uv publish --token "$PYPI_TOKEN"        # or: uv publish   (trusted publishing)
```

With Trusted Publishing, this workflow replaces the manual upload:

```yaml
# .github/workflows/release.yml
name: release
on:
  push:
    tags: ["v*"]

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write          # required for trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv build
      - run: uv publish
```

---

## 5. Verify the published package

Install from real PyPI, in a throwaway environment, and drive the server the way
a client will:

```bash
uvx --refresh tslab-mcp --help 2>/dev/null || true
npx @modelcontextprotocol/inspector uvx tslab-mcp
```

The Inspector should list eight `tsf_*` tools with populated schemas. The base
install is ~340 MB and starts quickly. Repeat with the extra —
`uvx --from 'tslab-mcp[foundation]' tslab-mcp` — and expect several minutes and
roughly 2 GB the first time; that is TimeCopilot's dependency tree, not a hung
install. Both shapes are worth checking, since they are separate code paths:
`tsf_list_models` should report `foundation_extra_installed` accordingly.

---

## 6. After release

- **Filenames on PyPI are immutable.** You cannot re-upload `0.1.0`, even after
  deleting it. A mistake means `0.1.1`. This is exactly why step 3 exists.
- **Yanking** (`pip install` skips it, existing pins still resolve) is the remedy
  for a broken release: *Manage project → Releases → Yank*. Prefer yanking to
  deleting.
- **Bumping**: edit the single version source, commit, tag `vX.Y.Z`, push the
  tag. Keep the tag and the metadata identical.
- Re-run the §0 checks on every TimeCopilot upgrade, and confirm the parity
  test still passes: it is the thing standing between "the backends agree" and
  a claim nobody has verified.
