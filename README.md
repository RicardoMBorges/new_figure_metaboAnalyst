# new_figure_metaboAnalyst


# MetaboAnalyst → Post-Plots (PCA, PLS-DA, VIP, R²/Q²)

Minimal Streamlit app to render PCA/PLS-DA score plots with outline-only ellipses,
VIP Top-N, and R²/Q² cross-validation computed from `data_normalized.csv`
(with “Label” row).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py


#  MetaboAnalyst → Post-Plots (PCA, PLS-DA, VIP, R²/Q²)

This Streamlit app helps create **publication-ready figures** from results exported from MetaboAnalyst.
You can generate **PCA and PLS-DA score plots**, **VIP Top-N barplots**, and **R²/Q² validation plots** directly from the CSVs—**no external metadata required**.

---

##  Features

*  **Automatic axis detection** (`PC1`, `Comp 1`, etc.)
*  Grouping from:

  * Existing column in score CSV
  * Parsing from sample names (regex)
  * Single group mode
*  90% confidence ellipses on PCA & PLS-DA
*  VIP Top-N slider (adjust number of features shown)
*  R²/Q² cross-validation computed directly from `data_normalized.csv` (Label row)
*  Export plots as **HTML** or **PNG**

---

##  Required Files

| File                  | Required | Description                                                        |
| --------------------- | -------- | ------------------------------------------------------------------ |
| `pca_score.csv`       | ✅        | PCA score table from MetaboAnalyst                                 |
| `plsda_score.csv`     | ✅        | PLS-DA score table from MetaboAnalyst                              |
| `plsda_vip.csv`       | ✅        | VIP scores table                                                   |
| `data_normalized.csv` | Optional | Feature × sample matrix; **row 2 must be “Label”** with group info |
| `plsda_r2q2.csv`      | Optional | External R²/Q² results (if not computing internally)               |

 **Important:**

* `data_normalized.csv` must have **samples as columns**.
* The **second row** (`Label`) should contain the group for each sample.
* Rows below that must be numeric data.

---

##  Installation (local)

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
.venv\Scripts\activate   # (Windows)
# source .venv/bin/activate  (Mac/Linux)

pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io).
3. Click **“New app”** → select repo, branch `main`, file `app.py`.
4. Optional: set Python 3.10.
5.  Deploy.

Your app will be live in minutes 🚀

---

##  Usage

1. Open the app locally or online.
2. Upload the required CSV files.
3. Choose your grouping method in the sidebar.
4. Adjust confidence levels, VIP Top-N, and R²/Q² parameters.
5. Download figures as PNG or HTML.

---

##  Example Workflow

```text
1. Export PCA/PLS-DA/VIP results from MetaboAnalyst
2. Upload pca_score.csv, plsda_score.csv, plsda_vip.csv
3. Optionally upload data_normalized.csv (for R²/Q²)
4. Customize plot appearance (confidence level, colors, Top-N)
5. Download publication-quality figures
```

## Tutorial Data

Tutorial data was downloaded from https://www.metaboanalyst.ca/MetaboAnalyst/upload/StatUploadView.xhtml
- Peak lists and intensity files for 50 urine samples measured by 1H NMR (Psihogios NG, et al.). Group 1- control; group 2 - severe kidney disease. 

---

##  requirements.txt

```txt
streamlit>=1.36
pandas>=2.0,<3
numpy>=1.24
plotly>=5.20
scipy>=1.10
scikit-learn>=1.3
kaleido>=0.2.1
```

---

##  Tips

* Use the **sidebar color mapping** to fix group colors.
* `kaleido` enables PNG export — but HTML download always works.
* R²/Q² will compute automatically if `data_normalized.csv` is correctly formatted.
* No need for `metadata.csv`.

---

##  License

MIT License © 2025

