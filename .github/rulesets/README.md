# rulesets

Versioned reference definitions for this repo's GitHub rulesets
(Settings > Rules > Rulesets). **These files are not auto-applied by
GitHub** — they're the source of truth you re-apply by hand after editing,
via `gh api`, so branch/tag protection is reviewable in PRs like everything
else in this repo.

## `branch-rules.json` — LIVE (ruleset id `20587475`, named `BranchRules`)

Covers both `main` and `develop`. Built via the GitHub UI, not via this
file originally — this file was written to match what's actually
configured, after the fact.

- **`required_signatures` is active.** `aganrandave` has SSH commit
  signing configured; `john-babalola1307`, `bolanle-ea`, `Oluwatiseunla`,
  and `Ololade-ajaegbu` still need to set theirs up before they can push —
  see [`docs/commit_signing_setup.md`](../../docs/commit_signing_setup.md).
  Until then, their pushes to `main`/`develop` (including via PR merge)
  will be rejected with "Commits must have verified signatures."
- **`require_code_owner_review` is active.** `.github/CODEOWNERS` is now
  enforced — PRs need an approval from the specific layer owner(s) listed
  there, not just any approver. The admin bypass (`bypass_actors`, see
  below) still lets `aganrandave` merge without waiting on that specific
  reviewer.

To reapply after editing this file:

```bash
gh api --method PUT -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" \
  repos/aganrandave/DAT608-POS-Fraud-Detection/rulesets/20587475 \
  --input .github/rulesets/branch-rules.json
```

## `tag-immutability-ruleset.json` — NOT YET APPLIED

Drafted but never created on GitHub. Blocks `update` and `deletion` on any
`refs/tags/*`, making release tags immutable once cut. Apply with:

```bash
gh api --method POST -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" \
  repos/aganrandave/DAT608-POS-Fraud-Detection/rulesets --input .github/rulesets/tag-immutability-ruleset.json
```
