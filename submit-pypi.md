# Publishing `tslab-mcp` to PyPI

Steps to get a release onto PyPI, plus the things about *this* package that make
a release go wrong. Work top to bottom; the pre-flight section is where the real
risk lives.

---

## 0. Two blockers to settle before the first release

### The version is declared twice

`pyproject.toml` and `src/tslab_mcp/__init__.py` each carry `0.1.0`
independently. Nothing keeps them in step, and `manifest.py` uses **both**:
`__version__` for the `tslab_mcp` key, and the installed distribution metadata
for the pinned package list. If they drift, an exported manifest reports two
different versions of the same package — in a package whose entire purpose is a
defensible lineage record.

Single-source it before publishing. Hatchling reads the version straight out of
the module:

```toml
# pyproject.toml
[project]
-version = "0.1.0"
+dynamic = ["version"]

+[tool.hatch.version]
+path = "src/tslab_mcp/__init__.py"
```

`src/tslab_mcp/__init__.py` then becomes the only place a release number is
edited. Verify with `uv build && tar -tzf dist/*.tar.gz | head -1` — the
directory name carries the version.

### `requires-python = ">=3.10"` has only ever been tested on 3.13

The project declares support for 3.10–3.13 and ships classifiers to match, but
every test run, lint pass and manual verification in this repo has happened on
3.13. On 3.10–3.12 TimeCopilot pulls `tabpfn-time-series`, which pins
`pandas<2.2`, so those interpreters get a *materially different* dependency set —
pandas 2.1 with its older offset aliases, plus two extra models.

Either test it or narrow the claim. To test:

```bash
uv run --python 3.10 --isolated pytest      # repeat for 3.11, 3.12
```

To narrow instead, set `requires-python = ">=3.13"` and drop the 3.10/3.11/3.12
classifiers. Publishing an untested compatibility claim is the kind of thing
that generates issues you cannot reproduce.

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

Inspect what you are actually shipping — the wheel should contain the ten
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
dependencies, and this package needs TimeCopilot and its ~290 transitive
packages. A plain `pip install -i https://test.pypi.org/simple tslab-mcp` will
fail to resolve. To test the install path, pull the package from TestPyPI and
its dependencies from real PyPI:

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

The Inspector should list seven `tsf_*` tools with populated schemas. Expect the
first run to take several minutes and roughly 2.3 GB of downloads — that is
TimeCopilot's dependency tree, not a hung install. Worth confirming the README
sets that expectation before people meet it unprepared.

---

## 6. After release

- **Filenames on PyPI are immutable.** You cannot re-upload `0.1.0`, even after
  deleting it. A mistake means `0.1.1`. This is exactly why step 3 exists.
- **Yanking** (`pip install` skips it, existing pins still resolve) is the remedy
  for a broken release: *Manage project → Releases → Yank*. Prefer yanking to
  deleting.
- **Bumping**: edit the single version source, commit, tag `vX.Y.Z`, push the
  tag. Keep the tag and the metadata identical.
- Re-run the §0 compatibility question on every TimeCopilot upgrade. Its
  dependency pins move, and they are what constrain this package's supported
  Python and pandas range.
