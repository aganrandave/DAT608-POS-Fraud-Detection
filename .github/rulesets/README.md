# rulesets

Versioned reference definitions for this repo's GitHub rulesets
(Settings > Rules > Rulesets). **These files are not auto-applied by
GitHub** — they're the source of truth you re-apply by hand after editing,
via `gh api`, so branch/tag protection is reviewable in PRs like everything
else in this repo.

## `branch-rules.json` — LIVE (ruleset id `20587475`, named `BranchRules`)

Covers both `main` and `develop`. Built via the GitHub UI, not via this
file originally — this file was written to match what's actually
configured, after the fact. Known gaps against best practice, tracked
deliberately rather than silently:

- **`required_signatures` is currently omitted.** It was enabled once,
  which blocked the very first push to `main` (no team member has commit
  signing set up yet). Re-add it once SSH/GPG signing is configured for
  `aganrandave`, `john-babalola1307`, `bolanle-ea`, and `Oluwatiseunla` —
  see the root README or ask for the setup steps.
- **`require_code_owner_review` is `false`.** `.github/CODEOWNERS` exists
  but isn't currently enforced by this ruleset — any approver satisfies
  the PR gate, not specifically the layer owner. Flip to `true` once the
  team is comfortable with CODEOWNERS routing.

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
