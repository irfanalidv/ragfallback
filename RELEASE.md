# Release process

ragfallback publishes to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) —
GitHub Actions authenticates to PyPI with a short-lived OIDC token, so no
PyPI API token is ever stored as a GitHub secret. This replaces the
previous manual `twine upload` flow.

## One-time setup (do this once, on PyPI)

1. Log in at [pypi.org](https://pypi.org) as the `ragfallback` maintainer.
2. Go to **ragfallback → Manage → Publishing**.
3. Add a new trusted publisher:
   - Owner: `irfanalidv`
   - Repository name: `ragfallback`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
4. Save. No token, no secret — PyPI now trusts this exact GitHub Actions
   workflow running in this exact environment.
5. (Optional but recommended) Under the repo's **Settings → Environments**,
   create an environment named `pypi` and add yourself as a required
   reviewer. This adds a manual approval click before every publish —
   cheap insurance against an accidental tag push.

## Cutting a release

```bash
# 1. Bump the version in two places (must match exactly)
#    - pyproject.toml          -> [project] version = "2.3.0"
#    - ragfallback/__init__.py -> __version__ = "2.3.0"

# 2. Update CHANGELOG.md — move [Unreleased] items under a new
#    ## [2.3.0] - YYYY-MM-DD heading

# 3. Commit, tag, push
git add pyproject.toml ragfallback/__init__.py CHANGELOG.md
git commit -m "release: v2.3.0"
git tag v2.3.0
git push origin main --tags
```

Pushing the tag triggers `.github/workflows/release.yml`, which builds the
sdist + wheel, checks the tag matches `pyproject.toml`, and publishes to
PyPI. You can also trigger the same workflow by drafting a GitHub Release
from the tag — that's the better option if you want release notes attached
(GitHub will copy the matching CHANGELOG section in automatically if you
use "Generate release notes").

## If something goes wrong

- **Tag/version mismatch**: the workflow fails before publishing — fix the
  version, delete the tag (`git tag -d v2.3.0 && git push origin :v2.3.0`),
  and re-tag.
- **Need to yank a bad release**: do this on PyPI directly (Manage →
  Releases → Yank). Trusted Publishing has no effect on yanking.
- **Local manual publish (emergency only)**: `python -m build && twine
  upload dist/*` still works if you have a scoped PyPI API token, but
  prefer the workflow — it's what the README badge claims is happening.
