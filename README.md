# TransitText: Public Transport Feedback Classification

**Competition:** NSUCEC NeuraHack 2026 — Mock Preliminary Round  
**Team Name:** `[Your_Team_Name]` *(replace with your team name when creating `NeuraHack_Mock_Preli_<Team_Name>.zip`)*  

---

## 1. Approach Summary

To solve 6-class commuter feedback classification (`0: Delay`, `1: Fare`, `2: Safety`, `3: Cleanliness`, `4: Service`, `5: Other`) across **English, Bangla Unicode, Banglish, and mixed-language messages** using only 60 labelled examples (`sample.csv`), our reproducible pipeline combines three pillars:

1. **Rule-Compliant Domain Supervision & Noise Augmentation:**
   - Constructed a balanced **300-example trilingual synthetic corpus** (`50` examples/class across English, Bangla Unicode, and Banglish) derived strictly from the Section 2 category specifications, programmatically verified to have **zero overlap** with `test.csv` or `sample.csv`.
   - Applied **4x reproducible augmentation** (conversational greeting/closing wrappers, random character typos, and case variations) uniformly across all 6 classes so classifiers learn invariance to misleading discourse fillers and noisy spelling.
   - Built a **Dual-View representation** (`0.45 * raw + 0.55 * core_clause`) paired with soft-phonetic Banglish normalization (`bh->v`, `ph->f`, `sh->s`, `ee->i`, `oo->u`) for lexical matching.

2. **Rigorous Leak-Free Multi-Split Validation:**
   - Evaluated every pipeline with identical **5-Fold Stratified CV** on `sample.csv` (all vectorizers, SVD transformers, augmentations, and class centroids fit strictly inside each training fold).
   - Measured robustness across **Clean OOF**, **Seen-Wrapper Noisy OOF**, **Unseen-Wrapper Noisy OOF**, and **Zero-Shot Synthetic-Only -> Real** evaluations alongside 95% bootstrap confidence intervals and script-slice error analysis.

3. **5-Paradigm / 7-Model Ensemble (`All7Diverse`) + Multi-Seed Bagging:**
   - **Dense Multilingual Linear & SVM:** `LogisticRegression` (`C=6.0`) and 3-fold `CalibratedClassifierCV(LinearSVC)` on 768-d `Xenova/multilingual-e5-base` embeddings.
   - **Neural Manifold-Mixup Residual MLP:** PyTorch GELU Residual MLP with LayerNorm, Dropout, label smoothing (`0.05`), and Manifold Mixup on concatenated `E5-Base (768-d) + E5-Small (384-d)` embeddings.
   - **Multi-Channel Lexical Subword Models:** `LogisticRegression` and `ComplementNB` over word (`1–2`), character (`2–5`), and soft-phonetic character TF-IDF n-grams.
   - **Joint Multimodal SVD-Stacking:** `LogisticRegression` on fused `[E5-Base, E5-Small, TruncatedSVD(TF-IDF, 160)]`.
   - **Gradient Boosted Trees:** `LightGBM` over `TruncatedSVD` features and cosine similarities to training class centroids.
   - **Final Test Predictions:** 50% 5-Fold CV ensemble probabilities + 50% 3-seed full-data ensemble probabilities (`submission.csv`).

---

## 2. External Models & Resources Disclosed (Section 6, Rule 7)

All external resources are publicly available, open-weight, and require **no API keys**:

- **Pretrained Models (HuggingFace Hub, INT8 ONNX):**
  - [`Xenova/multilingual-e5-base`](https://huggingface.co/Xenova/multilingual-e5-base) (`onnx/model_int8.onnx`)
  - [`Xenova/multilingual-e5-small`](https://huggingface.co/Xenova/multilingual-e5-small) (`onnx/model_int8.onnx`)
- **Open-Source Python Libraries:** `numpy`, `pandas`, `scipy`, `scikit-learn`, `lightgbm`, `torch`, `transformers`, `onnxruntime`, `huggingface_hub`, `matplotlib`, `seaborn`, `python-dotenv`.
- **Environment File (`.env`):** Included as an empty file because no external API keys are used.

---

## 3. Instructions for Running the Notebook

1. **Prerequisites:** Python 3.10+ (tested on Python 3.14.8 on Windows CPU) with the competition files (`sample.csv`, `test.csv`, `sample_submission.csv`) and `.env` in the same directory as `notebook.ipynb`.
2. **Install Dependencies (recommended inside a virtual environment):**
   ```bash
   python -m venv .venv
   # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   # Linux/macOS:        source .venv/bin/activate
   pip install numpy pandas scipy scikit-learn lightgbm torch transformers onnxruntime huggingface_hub matplotlib seaborn python-dotenv nbformat nbconvert ipykernel
   ```
3. **Execute Reproducibly:**
   - Open `notebook.ipynb` in Jupyter / VS Code and **Run All Cells**, or execute from CLI:
     ```bash
     python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebook.ipynb
     ```
   - On the first run, `Xenova/multilingual-e5-base` (~278 MB) and `Xenova/multilingual-e5-small` (~118 MB) download automatically from HuggingFace Hub and cache locally. Subsequent runs finish in ~2.5 minutes on CPU and regenerate `submission.csv`.

---

## 4. Submission Package Checklist

When creating `NeuraHack_Mock_Preli_<Team_Name>.zip`, include **only** the four required deliverables:
- `submission.csv`
- `notebook.ipynb`
- `README.md`
- `.env`
