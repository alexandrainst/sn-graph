# Releasing

The version is not stored in the repository. `pyproject.toml` carries
`version = "0.0.0"` as a placeholder; the real version is derived from the git
tag at build time. The tag is the single source of truth.

## Current version

```bash
git describe --tags --abbrev=0     # latest release
git tag -l --sort=-v:refname       # full history
```

## Cut a release

1. Make sure `main` is green and contains everything you want to ship.

2. Tag it, with a `v` prefix and semantic versioning:

   ```bash
   git tag v0.4.0 && git push origin v0.4.0
   ```

   Pushing a tag runs no workflow. That is expected.

3. Create the GitHub Release. Publishing it runs
   `.github/workflows/release.yml`, which runs the tests, builds the
   distributions, checks the built version matches the tag, and uploads to
   PyPI.

   ```bash
   gh release create v0.4.0 --title "..." --notes "..."
   ```

   Add `--draft` to write the notes now and publish later - a draft triggers
   nothing. Skip it if you are happy to publish straight away.

The release notes are the project's changelog; there is no `CHANGELOG.md`.

## Local builds

Between releases the version resolves to something like
`0.3.0.post36.dev0+d534de8` - 36 commits past `v0.3.0`. That is normal.

Poetry installs the versioning plugin itself on `poetry install`, declared in
`[tool.poetry.requires-plugins]`. There is nothing to set up by hand.

## Credentials

PyPI publishing is configured outside this repository, so no token is stored
here. If the publish step fails to authenticate, that configuration is what
needs attention.
