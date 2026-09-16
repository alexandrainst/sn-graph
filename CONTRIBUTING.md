# Contributing

Bug reports, improvements, and pull requests are welcome. Open a GitHub issue
to report a bug or suggest a change. If you already have a fix, open a pull
request.

## For maintainers

### Releasing

The version is not stored in the repository. `pyproject.toml` carries
`version = "0.0.0"` as a placeholder; the real version is derived from the git
tag at build time. The tag is the single source of truth.

#### Current version

```bash
git describe --tags --abbrev=0     # latest release
git tag -l --sort=-v:refname       # full history
```

#### Prepare a release

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

   The tag does not need to exist: it is created and arrives locally with your next `git pull`. Tag beforehand only if you need to pin the release to some other commit.

The release notes are the project's changelog; there is no `CHANGELOG.md`.

#### Credentials

Publishing uses PyPI [Trusted
Publishing](https://docs.pypi.org/trusted-publishers/).

### Documentation

The notebooks need a Poetry environment with Jupyter installed. Set it up with:

```bash
make install
```

Then rebuild and deploy the documentation:

```bash
make mkdocs
```

This runs the tutorial notebooks, builds the MkDocs site, and deploys it to
GitHub Pages. When it finishes, check the
[published documentation](https://alexandrainst.github.io/sn-graph/).
