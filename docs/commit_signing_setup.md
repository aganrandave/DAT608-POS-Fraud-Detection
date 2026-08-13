# Commit Signing Setup

The `main`/`develop` branch ruleset requires every commit to carry a
verified signature (`required_signatures`, see
[`.github/rulesets/branch-rules.json`](../.github/rulesets/branch-rules.json)).
Once this is enforced, an unsigned push will be rejected outright — set
this up **before** you need to push, not after.

SSH signing is the fastest path since you already have SSH keys for Git.
GPG works too if you already have a GPG setup you prefer.

## SSH signing (recommended)

1. Generate a key dedicated to commit signing (don't reuse your GitHub
   auth/clone key):

   ```bash
   ssh-keygen -t ed25519 -C "<your-github-username>-commit-signing" -f ~/.ssh/git_signing_ed25519
   ```

2. Tell Git to sign every commit with it:

   ```bash
   git config --global gpg.format ssh
   git config --global user.signingkey ~/.ssh/git_signing_ed25519.pub
   git config --global commit.gpgsign true
   ```

3. Register the **public** key with GitHub as a signing key (not an auth
   key — it's a separate list under Settings > SSH and GPG keys):

   ```bash
   gh auth refresh -h github.com -s admin:ssh_signing_key
   gh ssh-key add ~/.ssh/git_signing_ed25519.pub --type signing --title "<your-username> commit signing"
   ```

   No `gh` CLI? Copy `~/.ssh/git_signing_ed25519.pub` and paste it into
   GitHub under Settings > SSH and GPG keys > New SSH key > Key type:
   **Signing Key**.

4. Verify: make a commit, then check it shows "Verified" on GitHub, or
   locally with:

   ```bash
   git log --show-signature -1
   ```

## GPG signing (alternative)

```bash
gpg --full-generate-key   # RSA 4096 or ed25519, your GitHub email as the identity
gpg --list-secret-keys --keyid-format=long
git config --global user.signingkey <KEY_ID>
git config --global commit.gpgsign true
gpg --armor --export <KEY_ID>   # paste this into GitHub Settings > SSH and GPG keys > New GPG key
```

## Notes

- These `git config --global` settings apply to every repo on your
  machine, not just this one — that's the standard/recommended setup.
- If you already had commits queued up before doing this step, re-sign them
  with `git commit --amend -S --no-edit` (last commit) or an interactive
  rebase (`git rebase --exec 'git commit --amend --no-edit -S' -i <base>`)
  for a range — only necessary for commits not yet merged into `main`.
