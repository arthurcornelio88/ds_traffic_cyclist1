import os
import json
import uuid
from typing import Literal, Optional
from urllib.request import urlopen
import app.app_config as _  # forcer le sys.path side effect
from app.classes import RFPipeline, NNPipeline



def update_summary(
    summary_path: str,
    model_type: str,
    run_id: str,
    rmse: float,
    r2: float,
    model_uri: str,
    env: str = "prod",
    test_mode: bool = False
):
    import datetime

    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_type": model_type,
        "env": env,
        "test_mode": test_mode,
        "run_id": run_id,
        "rmse": rmse,
        "r2": r2,
        "model_uri": model_uri
    }

    # Déterminer le chemin local
    summary_path_local = "/tmp/summary.json" if summary_path.startswith("gs://") else summary_path
    summary = []

    # Télécharger ou charger si existant
    if summary_path.startswith("gs://"):
        os.system(f"gsutil cp {summary_path} {summary_path_local} || touch {summary_path_local}")
    if os.path.exists(summary_path_local):
        with open(summary_path_local, "r") as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ summary.json vide ou corrompu. Réinitialisation.")
                summary = []

    # ⛔ Supprimer anciens du même type/env/test_mode
    summary = [
        s for s in summary
        if not (
            s["model_type"] == model_type
            and s["env"] == env
            and s["test_mode"] == test_mode
        )
    ]

    summary.append(entry)

    with open(summary_path_local, "w") as f:
        json.dump(summary, f, indent=2)

    if summary_path.startswith("gs://"):
        os.system(f"gsutil cp {summary_path_local} {summary_path}")
        print(f"✅ summary.json mis à jour et uploadé vers {summary_path}")
    else:
        print(f"✅ summary.json mis à jour localement : {summary_path}")

# === Chargement du meilleur modèle depuis le résumé
def get_best_model_from_summary(
    model_type: str,
    summary_path: str,
    env: str = "prod",
    metric: Literal["rmse", "r2"] = "rmse",
    test_mode: Optional[bool] = False
):
    if summary_path.startswith("gs://"):
        summary = _read_gcs_json(summary_path)
    elif summary_path.startswith("http"):
        with urlopen(summary_path) as f:
            summary = json.load(f)
    else:
        with open(summary_path, "r") as f:
            summary = json.load(f)

    filtered = [
        r for r in summary
        if r["model_type"] == model_type
        and r["env"] == env
        and r["test_mode"] == test_mode
        #and r["rmse"] > 0  # éviter les modèles fictifs/perfectibles
    ]

    if not filtered:
        raise RuntimeError(f"Aucun modèle trouvé pour type={model_type}, env={env}, test_mode={test_mode}")

    # Tri combiné : r2 décroissant puis rmse croissant
    best = max(filtered, key=lambda r: (r["r2"], -r["rmse"]))

    print(f"✅ Modèle {model_type} sélectionné : {best['run_id']} (r2={best['r2']}, rmse={best['rmse']})")

    local_model_path = _download_gcs_dir(best["model_uri"], prefix=model_type)

    # 💡 Si un sous-dossier "rf" ou "nn" existe à l'intérieur, on l'utilise
    subfolder = os.path.join(local_model_path, model_type)
    if os.path.isdir(subfolder):
        local_model_path = subfolder

    if model_type == "rf":
        return RFPipeline.load(local_model_path)
    elif model_type == "nn":
        return NNPipeline.load(local_model_path)
    else:
        raise ValueError("Type de modèle non reconnu")


# === Téléchargement GCS vers /tmp
def _download_gcs_dir(gcs_uri: str, prefix="model") -> str:
    from google.cloud import storage

    bucket_name, path = gcs_uri.replace("gs://", "").split("/", 1)
    local_tmp_dir = f"/tmp/{prefix}_{uuid.uuid4().hex}"
    os.makedirs(local_tmp_dir, exist_ok=True)

    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=path))

    for blob in blobs:
        rel_path = os.path.relpath(blob.name, path)
        local_file = os.path.join(local_tmp_dir, rel_path)
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)

    return local_tmp_dir


# === Lecture JSON depuis GCS
def _read_gcs_json(gs_path: str) -> dict:
    from google.cloud import storage

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise EnvironmentError("Variable GOOGLE_APPLICATION_CREDENTIALS non définie pour accéder à GCS")

    bucket_name, blob_path = gs_path.replace("gs://", "").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    return json.loads(blob.download_as_text())
