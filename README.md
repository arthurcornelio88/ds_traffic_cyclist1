# 🚲 Bike Count Prediction (Streamlit + MLflow + GCS)

This project predicts **hourly bicycle traffic** from Paris sensors using a simple UI powered by **Streamlit**.

Key features:
- 🧠 Predict using **Random Forest** or **Neural Network**
- ☁️ Fully hosted on **Streamlit Cloud** + **GCP** (no local server needed)
- 📦 Custom model registry via `summary.json` on GCS
- 🔁 Batch or single prediction, CSV download

---

## 🛠 Architecture Overview

```mermaid
graph TD
  A[User Input / CSV] --> B[Streamlit App]
  B --> C[Load best model from summary.json]
  C --> D[Model files on GCS]
  B --> E[Prediction UI]

  subgraph ML Training
    F[train.py]
    F --> G[Fit + Save model locally]
    G --> H[Upload to GCS]
    G --> I[Update summary.json]
  end
````

---

## 🧠 DIY Model Registry (no MLflow dependency at runtime)

This project **does not rely on the MLflow model registry**.

✅ Instead:

* We log metadata to a lightweight **`summary.json`** file on GCS
* We track only the best model (by `rmse` or `r2`) per type/env/test mode
* We load the corresponding GCS model directory at runtime

This allows for:

* Zero MLflow runtime dependencies
* Easy GCS integration
* Fast cold starts on **Streamlit Cloud**

---

## ✅ Run on Streamlit Cloud

Works out-of-the-box. Here's how to deploy:

### Step-by-step

1. **Push to GitHub** (make sure `requirements.txt` exists)
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Create a new app pointing to `app/streamlit_app.py`
4. Go to **"Secrets"**, and paste your GCP service account JSON:

   ```toml
   [gcp_service_account]
   # Paste the raw JSON here
   ```
5. Done ✅ — your model will auto-load from GCS and predictions are served live.

---

## 🧪 Local Development

### Prereqs

```hcl
uv init
uv venv
uv sync
```

### Train & Upload Models

```bash
# Dev mode, small dataset
python app/train.py --env dev --model_test

# Full training + upload to GCS
python app/train.py --env prod
```

### Launch Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

---

## ☁️ Optional: MLflow Local UI

Only used during training (not needed at runtime):

```bash
# Local backend
mkdir -p mlruns/artifacts

# Prod (optional)
export GOOGLE_APPLICATION_CREDENTIALS=./mlflow-ui-access.json

mlflow server \
  --backend-store-uri file:./mlruns \
  --default-artifact-root gs://df_traffic_cyclist1/mlruns \
  --host 127.0.0.1 \
  --port 5000
```

---

Perfect — let's finalize your `README.md` with a full GCP service account setup section, documenting the **three key service accounts** used in your project:

---

## 🔐 GCP Service Accounts Setup

Your project uses **three separate service accounts** for clear roles separation between training, inference, and UI.

### 1. `mlflow-trainer@...`

Used in `train.py` to:

* Upload trained models to GCS
* Update `summary.json`

**Required Roles:**

* `Storage Object Admin`

➡️ Set the credentials locally:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./mlflow-trainer.json
```

---

### 2. `mlflow-ui-access@...`

Used **only** if you run MLflow UI in **prod mode** with artifact logging to GCS.

**Required Roles:**

* `Storage Object Viewer`

➡️ Example:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./mlflow-ui-access.json

mlflow server \
  --backend-store-uri file:./mlruns \
  --default-artifact-root gs://df_traffic_cyclist1/mlruns \
  --host 127.0.0.1 --port 5000
```

---

### 3. `gcp_service_account` (for Streamlit Cloud)

Used by your **deployed Streamlit app** to:

* Download the latest model artifacts
* Read `summary.json` from GCS

**Required Role:**

* `Storage Object Viewer`

#### 👉 Add to Streamlit secrets:

In [streamlit.io/cloud](https://streamlit.io/cloud) → *Secrets*:

```toml
[gcp_service_account]
# Paste your full JSON key here
```

---

## ✅ Permissions Recap

| Service Account       | Purpose               | Permissions             |
| --------------------- | --------------------- | ----------------------- |
| `mlflow-trainer`      | Train + upload models | `Storage Object Admin`  |
| `mlflow-ui-access`    | MLflow UI (prod)      | `Storage Object Viewer` |
| `gcp_service_account` | Streamlit Cloud app   | `Storage Object Viewer` |

All three can be created from the GCP IAM console:

> [https://console.cloud.google.com/iam-admin/serviceaccounts](https://console.cloud.google.com/iam-admin/serviceaccounts)

---

## 📂 Directory Structure

```
app/
├── app_config.py              # sys.path trick
├── streamlit_app.py           # UI
├── train.py                   # Training CLI
├── model_registry_summary.py  # DIY model registry
├── classes.py                 # Pipeline logic (RF/NN)
data/
models/
```

---

## ✅ Summary

* ✔️ Fully serverless inference using Streamlit Cloud + GCS
* ✔️ DIY model registry using `summary.json`
* ✔️ Streamlit picks the best model automatically
* ✔️ No MLflow required at runtime
* ⚡ Cold-start ready + fast batch predictions

---

## 👨‍🔬 Authors

Built by [Arthur Cornélio](https://github.com/arthurcornelio88), [Ibtihel Nemri]() et [Bruno Happi]().

