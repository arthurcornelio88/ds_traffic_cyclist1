# mlflow_model_loader.py

import tempfile
import os
from classes import RFPipeline, NNPipeline
from mlflow.tracking import MlflowClient

def get_best_run_id(model_type: str, env: str = "dev", test_mode: bool = False, metric: str = "", ascending=True):
    client = MlflowClient()
    experiment = client.get_experiment_by_name("traffic_cycliste_experiment")

    filter_str = f"tags.mode = '{env}' and tags.test_mode = '{str(test_mode)}'"

    metric_key = metric or ("rf_rmse_train" if model_type == "rf" else "nn_rmse_train")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_str,
        order_by=[f"metrics.{metric_key} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )

    if not runs:
        raise RuntimeError(f"Aucun modèle {model_type} trouvé pour env={env} test_mode={test_mode}")

    return runs[0].info.run_id


def load_pipeline_from_mlflow(model_type: str, env: str = "dev", test_mode: bool = False):
    run_id = get_best_run_id(model_type, env, test_mode)
    artifact_subpath = "rf_model" if model_type == "rf" else "nn_model"

    client = MlflowClient()
    temp_dir = tempfile.mkdtemp()
    client.download_artifacts(run_id, artifact_subpath, temp_dir)

    prefix = os.path.join(temp_dir, artifact_subpath, "rf" if model_type == "rf" else "nn")

    if model_type == "rf":
        return RFPipeline.load(prefix)
    elif model_type == "nn":
        return NNPipeline.load(prefix)
    else:
        raise ValueError("Modèle inconnu")
