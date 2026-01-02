# Repository Reset Guide

**Date:** 2026-01-01

This guide will help you delete the current GitHub repository and create a new one with all your latest code.

## Current Status
✅ All local changes have been committed
✅ Latest commit: `3e79849` - "feat: add PyPI publishing workflow and update project configuration"
✅ Remote: `https://github.com/openmirlab/sheetsage-infer.git`

## Step-by-Step Instructions

### Step 1: Delete the Current GitHub Repository

1. Go to your GitHub repository: https://github.com/openmirlab/sheetsage-infer
2. Click on **Settings** (in the repository navigation bar)
3. Scroll down to the **Danger Zone** section at the bottom
4. Click **Delete this repository**
5. Type the repository name `openmirlab/sheetsage-infer` to confirm
6. Click **I understand the consequences, delete this repository**

⚠️ **Warning**: This will permanently delete the repository and all its issues, pull requests, and wiki content. Make sure you have everything you need locally.

### Step 2: Create a New GitHub Repository

1. Go to https://github.com/organizations/openmirlab/repositories/new
   (Or go to https://github.com/new if creating under your personal account)
2. Fill in the repository details:
   - **Repository name**: `sheetsage-infer`
   - **Description**: `Minimal inference-only version of SheetSage for music transcription`
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click **Create repository**

### Step 3: Update Remote and Push Code

After creating the new repository, run these commands:

```bash
cd /home/mku666/2025/chord_analysis/sheetsage-infer-latest/sheetsage-infer

# Remove the old remote
git remote remove origin

# Add the new remote (replace with your new repository URL if different)
git remote add origin https://github.com/openmirlab/sheetsage-infer.git

# Verify the remote
git remote -v

# Push all branches and tags to the new repository
git push -u origin master --force

# If you have tags, push them too
git push origin --tags
```

### Step 4: Verify Everything is Pushed

1. Visit your new repository on GitHub
2. Verify that:
   - All files are present
   - The commit history is intact
   - The `.github/workflows/publish.yml` file exists
   - The `pyproject.toml` has your latest changes

### Step 5: Set Up PyPI Trusted Publishing (Optional but Recommended)

If you want to use the PyPI publishing workflow:

1. Go to https://pypi.org/manage/account/publishing/
2. Click **Add a new pending publisher**
3. Fill in:
   - **PyPI project name**: `sheetsage-infer`
   - **Owner**: `openmirlab` (or your GitHub username)
   - **Repository name**: `sheetsage-infer`
   - **Workflow filename**: `publish.yml`
4. Click **Add pending publisher**
5. Approve the publisher on GitHub when prompted

## Alternative: Keep Same Repository Name

If you want to keep the same repository name and just reset it:

1. Delete the repository (Step 1 above)
2. Create a new repository with the **same name** (Step 2 above)
3. Follow Step 3 to push your code

## Troubleshooting

### If you get authentication errors:
```bash
# Use GitHub CLI if installed
gh auth login

# Or use a personal access token
git remote set-url origin https://YOUR_TOKEN@github.com/openmirlab/sheetsage-infer.git
```

### If you want to preserve issues/PRs:
- Export issues before deleting (GitHub Settings → Export repository data)
- Or use GitHub's repository transfer feature instead of deleting

