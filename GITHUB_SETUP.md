# GitHub Setup

## Recommended repository

Repository name: `coffee-shop-demand-forecasting`

Suggested description:

> End-to-end Python demand forecasting project using POS-style data, data quality cleanup, feature engineering, time-aware validation, and tree-based regression models.

Recommended visibility: **Public**

## Create the repository on GitHub

1. Go to GitHub and sign in or create an account.
2. Choose **New repository**.
3. Name it `coffee-shop-demand-forecasting`.
4. Set it to **Public**.
5. Do not initialize it with a README, `.gitignore`, or license because this project already includes them.
6. Create the repository.

## Upload with GitHub's web interface

For the simplest first upload:

1. Open the new empty repository.
2. Choose **uploading an existing file**.
3. Unzip the portfolio package on your computer.
4. Drag the *contents* of the `coffee-shop-demand-forecasting` folder into GitHub, preserving the folders.
5. Commit with a message such as `Add demand forecasting portfolio project`.

## Or upload with Git from Terminal

From inside the unzipped project folder:

```bash
git init
git add .
git commit -m "Add demand forecasting portfolio project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/coffee-shop-demand-forecasting.git
git push -u origin main
```

## After upload

- Confirm the README renders correctly on the repository home page.
- Open the notebook on GitHub and confirm its outputs display.
- Check that `data/sample_pos_transactions.csv` is the only dataset in the repository.
- Make sure the original private POS CSV is never committed.
- Pin the repository to your GitHub profile.
- Add your GitHub profile URL to your resume and job application once the repo is live.
