# 👁 RetinaScope — Deployment Guide

Diabetic Retinopathy Detection via Retinal Fundus Images  
**Backend:** Streamlit Community Cloud · **Frontend:** GitHub Pages

---

## 📁 Repository Structure

```
retinascope/
├── app.py                ← Streamlit app (backend + UI)
├── requirements.txt      ← Python dependencies
├── dr_model.keras        ← Your trained model (YOU MUST ADD THIS)
├── index.html            ← GitHub Pages landing page (frontend)
└── README.md
```

---

## 🚀 Step 1 — Set Up Your GitHub Repository

1. Go to [github.com](https://github.com) → **New Repository**
2. Name it: `retinascope` (or any name you prefer)
3. Set visibility: **Public** (required for free GitHub Pages)
4. Click **Create repository**

---

## 📤 Step 2 — Push All Files

Open your terminal in the folder containing these files:

```bash
git init
git add .
git commit -m "Initial commit — RetinaScope DR Detection"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/retinascope.git
git push -u origin main
```

> ⚠️ Replace `YOUR-USERNAME` with your actual GitHub username.

---

## 🧠 Step 3 — Add Your Trained Model

Your trained model file is `dr_model.keras`.  
Copy it into the repo folder before pushing:

```bash
cp /path/to/dr_model.keras ./dr_model.keras
git add dr_model.keras
git commit -m "Add trained MobileNetV2 model"
git push
```

> 📌 **Note:** If your model is >100 MB, use [Git LFS](https://git-lfs.github.com):
> ```bash
> git lfs install
> git lfs track "*.keras"
> git add .gitattributes
> git commit -m "Track large model with Git LFS"
> git push
> ```

---

## ☁️ Step 4 — Deploy Backend on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **New app**
4. Fill in:
   - **Repository:** `YOUR-USERNAME/retinascope`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**

Streamlit will install dependencies from `requirements.txt` automatically.

Once deployed, your app URL will be:
```
https://YOUR-USERNAME-retinascope.streamlit.app
```

---

## 🌐 Step 5 — Enable GitHub Pages (Frontend)

1. Go to your repo on GitHub
2. Click **Settings** → **Pages** (in the left sidebar)
3. Under **Source**, select:
   - Branch: `main`
   - Folder: `/ (root)`
4. Click **Save**

Your landing page will be live at:
```
https://YOUR-USERNAME.github.io/retinascope/
```

---

## 🔗 Step 6 — Connect Frontend to Backend

Open `index.html` and replace **all 3 instances** of:
```
https://YOUR-APP.streamlit.app
```
with your actual Streamlit URL, e.g.:
```
https://johndoe-retinascope.streamlit.app
```

Also update the GitHub link:
```
https://github.com/YOUR-USERNAME/retinascope
```

Then commit and push:
```bash
git add index.html
git commit -m "Link GitHub Pages to Streamlit app"
git push
```

---

## ✅ Final Checklist

| Step | Task | Status |
|------|------|--------|
| 1 | GitHub repo created | ⬜ |
| 2 | Files pushed to `main` branch | ⬜ |
| 3 | `dr_model.keras` added to repo | ⬜ |
| 4 | Streamlit app deployed & live | ⬜ |
| 5 | GitHub Pages enabled | ⬜ |
| 6 | `index.html` URLs updated | ⬜ |

---

## 🛠 Local Development

To run the Streamlit app locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 📋 Requirements

| Package | Version |
|---------|---------|
| streamlit | ≥ 1.35.0 |
| tensorflow | ≥ 2.19.0 |
| opencv-python-headless | ≥ 4.9.0 |
| numpy | ≥ 1.26.0 |
| Pillow | ≥ 10.0.0 |
| scikit-learn | ≥ 1.4.0 |

---

## ⚠️ Clinical Disclaimer

RetinaScope is intended for **research and screening assistance only**.  
Results must not replace professional ophthalmological evaluation.  
All findings must be reviewed and confirmed by a qualified medical practitioner.

---

*RetinaScope · MobileNetV2 · APTOS 2019 · TensorFlow / Keras*
