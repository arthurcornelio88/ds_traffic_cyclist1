import os
import json
import datetime
from typing import Literal, Optional
from urllib.request import urlopen
import mlflow.pyfunc


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
    """
    Ajoute un enregistrement dans summary.json.
    """
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

    # Local file path
    if summary_path.startswith("gs://"):
        local_tmp = "/tmp/summary.json"
        os.system(f"gsutil cp {summary_path} {local_tmp} || touch {local_tmp}")
        summary_path_local = local_tmp
    else:
        summary_path_local = summary_path

    # Append to local summary
    summary = []
    if os.path.exists(summary_path_local):
        with open(summary_path_local, "r") as f:
            try:
                summary = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Fichier summary.json vide ou corrompu. Il sera réinitialisé.")

    summary.append(entry)

    with open(summary_path_local, "w") as f:
        json.dump(summary, f, indent=2)

    if summary_path.startswith("gs://"):
        os.system(f"gsutil cp {summary_path_local} {summary_path}")
        print(f"✅ summary.json mis à jour et uploadé vers {summary_path}")
    else:
        print(f"✅ summary.json mis à jour localement : {summary_path}")

import json
from urllib.request import urlopen
import mlflow.pyfunc

def get_best_model_from_summary(
    model_type: str,
    summary_path: str,
    metric: str = "rmse",
    env: str = "prod",
    test_mode: bool = False
):
    with urlopen(summary_path) as f:
        summary = json.load(f)

    filtered = [
        r for r in summary
        if r["model_type"] == model_type
        and r["env"] == env
        and r["test_mode"] == test_mode
    ]

    if not filtered:
        raise RuntimeError(f"Aucun modèle {model_type} trouvé dans le résumé")

    if metric == "rmse":
        best = min(filtered, key=lambda r: r["rmse"])
    elif metric == "r2":
        best = max(filtered, key=lambda r: r["r2"])
    else:
        raise ValueError("Métrique non supportée")

    print(f"✅ Chargement du modèle {model_type} avec {metric}={best[metric]} depuis {best['model_uri']}")
    return mlflow.pyfunc.load_model(best["model_uri"])