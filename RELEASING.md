# Releasing

The version is not stored in the repository. `pyproject.toml` carries
`version = "0.0.0"` as a placeholder; the real version is derived from the git
tag at build time. The tag is the single source of truth.

## Current version

```bash
git describe --tags --abbrev=0     # latest release
git tag -l --sort=-v:refname       # full history
```

## Prepare a release

1. Make sure `main` is green and contains everything you want to ship.

2. Create the GitHub Release. A Release is a GitHub object - a tag, plus a
   title and notes - and it is either a draft or published. Name the version
   with a `v` prefix, following semantic versioning:

   ```bash
   gh release create v0.4.0 --title "..." --notes "..."
   ```

   **Without `--draft`, this command publishes the package to PyPI.** Pass
   `--draft` to create the Release without triggering anything, then publish it
   later with `gh release edit v0.4.0 --draft=false` or from the Releases page.

   The tag does not need to exist: it is created for you, on the current tip of
   `main` **on GitHub** - not your local checkout, which need not even be on
   that branch - and arrives locally with your next `git pull`. Tag beforehand
   only if you need to pin the release to some other commit.

The release notes are the project's changelog; there is no `CHANGELOG.md`.

## Credentials

Publishing uses PyPI [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/).
