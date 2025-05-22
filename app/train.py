import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import os
from app.classes import RFPipeline, NNPipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

###
def setup_environment(env: str):
    if env == "dev":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("traffic_cycliste_experiment")
        data_path = "data/comptage-velo-donnees-compteurs.csv"
        artifact_path = "models/"
    elif env == "prod":
        # 1. Tracking URI en local (facultatif si juste log)
        mlflow.set_tracking_uri("file:/tmp/mlruns")
        mlflow.set_experiment("traffic_cycliste_experiment")

        # 2. Chemin des données + artefacts
        data_path = "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs.csv"
        artifact_path = "/tmp/models/"
        os.makedirs(artifact_path, exist_ok=True)

        # 3. Auth GCP (si via secrets.toml → ajuster selon ton contexte)
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
            gcp_credentials_path = "./gcp.json"  # ⚠️ À adapter
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path
            if not os.path.exists(gcp_credentials_path):
                raise FileNotFoundError(f"Fichier de clé GCP introuvable : {gcp_credentials_path}")

    else:
        raise ValueError("Environnement invalide : choisir 'dev' ou 'prod'")
    
    return data_path, artifact_path


def load_and_clean_data(path: str, sample_size: int = None):
    df = pd.read_csv(path, sep=";")
    df[['latitude', 'longitude']] = df['Coordonnées géographiques'].str.split(',', expand=True).astype(float)
    df_clean = df.dropna(subset=['latitude', 'longitude'])

    X = df_clean.drop(columns='Comptage horaire')
    y = df_clean['Comptage horaire']

    if sample_size:
        X_sample = X.sample(sample_size, random_state=42)
        y_sample = y.loc[X_sample.index]
        return X_sample, y_sample
    return X, y


def train_rf(X, y, artifact_path, env, test_mode):
    y = y.to_numpy()

    run_name = f"RandomForest_Train_{env}" + ("_TEST" if test_mode else "")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.sklearn.autolog(disable=True)

        rf = RFPipeline()
        rf.fit(X, y)

        mlflow.log_param("rf_n_estimators", rf.model.n_estimators)
        mlflow.log_param("rf_max_depth", rf.model.max_depth)

        y_pred = rf.predict(X)
        rmse_rf = np.sqrt(mean_squared_error(y, y_pred))
        r2_rf = r2_score(y, y_pred)

        mlflow.log_metric("rf_rmse_train", rmse_rf)
        mlflow.log_metric("rf_r2_train", r2_rf)

        print(f"🎯 Random Forest – RMSE : {rmse_rf:.2f} | R² : {r2_rf:.4f}")

        model_path = os.path.join(artifact_path, "rf_prod")
        os.makedirs(model_path, exist_ok=True)
        rf.save(os.path.join(model_path, "rf"))
        mlflow.log_artifacts(model_path, artifact_path="rf_model")


def train_nn(X, y, artifact_path, env, test_mode):
    y = y.to_numpy(dtype="float32")

    run_name = f"NeuralNet_Train_{env}" + ("_TEST" if test_mode else "")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mode", env)
        mlflow.set_tag("test_mode", test_mode)
        mlflow.log_metric("test_mode", int(test_mode))
        mlflow.tensorflow.autolog(disable=True)

        nn = NNPipeline()
        epochs = 50
        batch_size = 128
        nn.fit(X, y, epochs=epochs, batch_size=batch_size)

        mlflow.log_param("nn_epochs", epochs)
        mlflow.log_param("nn_batch_size", batch_size)
        mlflow.log_param("nn_embedding_dim", nn.embedding_dim)

        total_params = nn.model.count_params()
        mlflow.log_metric("nn_total_params", total_params)

        y_pred = nn.predict(X).flatten()
        rmse_nn = np.sqrt(mean_squared_error(y, y_pred))
        r2_nn = r2_score(y, y_pred)

        mlflow.log_metric("nn_rmse_train", rmse_nn)
        mlflow.log_metric("nn_r2_train", r2_nn)

        print(f"🎯 Neural Net – RMSE : {rmse_nn:.2f} | R² : {r2_nn:.4f} | Params: {total_params}")

        model_path = os.path.join(artifact_path, "nn_prod")
        os.makedirs(model_path, exist_ok=True)
        nn.save(os.path.join(model_path, "nn"))
        mlflow.log_artifacts(model_path, artifact_path="nn_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bike count models (RF + NN)")
    parser.add_argument('--model_test', action='store_true', help="Use 1000 samples for fast training")
    parser.add_argument('--env', default="dev", choices=["dev", "prod"], help="Select environment: dev or prod")
    args = parser.parse_args()

    sample_size = 1000 if args.model_test else None
    data_path, artifact_path = setup_environment(args.env)

    print(f"✅ Environnement {args.env} configuré : MLflow et artefacts prêts")
    print(f"✅ Chargement des données depuis {data_path}...")
    import pandas as pd
    df = pd.read_csv("gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs.csv", sep=";")
    print(df.head())


    X, y = load_and_clean_data(data_path, sample_size=sample_size)
    print(f"✅ Données chargées : {X.shape[0]} échantillons pour l'environnement {args.env}")

    train_rf(X, y, artifact_path, env=args.env, test_mode=args.model_test)
    train_nn(X, y, artifact_path, env=args.env, test_mode=args.model_test)

    print(f"✅ Modèles entraînés et loggés avec MLflow ({args.env})")
