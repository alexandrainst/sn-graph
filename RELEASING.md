# Releasing

The version is **not** stored in the repository. `pyproject.toml` deliberately
carries `version = "0.0.0"` as a placeholder; the real version is derived from
the git tag at build time by
[poetry-dynamic-versioning](https://github.com/mtkennerly/poetry-dynamic-versioning).
The tag is the single source of truth.

## Check the current version

```bash
git describe --tags --abbrev=0     # latest release
git tag -l --sort=-v:refname       # full history, newest first
```

## Cut a release

1. Make sure `main` is green and contains everything you want to ship.
2. Tag it, using a `v` prefix and semantic versioning:

   ```bash
   git tag v0.4.0 && git push origin v0.4.0
   ```

3. Create the GitHub Release. **Publishing the release is what publishes to
   PyPI**, so draft it first if you want to take your time over the notes:

   ```bash
   gh release create v0.4.0 --draft --title "..." --notes "..."
   ```

   A draft triggers nothing. Publishing it runs `.github/workflows/release.yml`,
   which runs the tests, builds the distributions, checks the built version
   matches the tag, and uploads to PyPI.

The release notes are the project's changelog - there is no `CHANGELOG.md`.

## Building locally

`python -m build` reads the git tag through the build backend and needs no
extra setup. Plain `poetry build` will report `0.0.0` unless you install the
plugin into Poetry itself:

```bash
poetry self add "poetry-dynamic-versioning[plugin]"
```

Between releases the derived version looks like `0.3.0.post35.dev0+a36c6ca`.
That is expected - it means the commit is 35 commits past the `v0.3.0` tag.

## One-time setup

PyPI publishing uses [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/), so no API token is
stored in the repository. It requires, once:

- On PyPI, under the project's *Publishing* settings, a GitHub publisher for
  owner `alexandrainst`, repository `sn-graph`, workflow `release.yml`,
  environment `pypi`.
- In the repository settings, an environment named `pypi`. Adding a required
  reviewer there gives a manual approval gate before anything is uploaded.
