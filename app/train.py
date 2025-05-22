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
def setup_environment(env: str, model_test: bool):
    if env == "dev":
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("traffic_cycliste_experiment")
        if model_test:
            data_path = "data/comptage-velo-donnees-compteurs_test.csv"
        else:
            data_path = "data/comptage-velo-donnees-compteurs.csv"
        artifact_path = "models/"
    elif env == "prod":
        # 1. Tracking URI en local (facultatif si juste log)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("traffic_cycliste_experiment")

        # 2. Chemin des données + artefacts
        if model_test:
            data_path = "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs_test.csv"
        else:
            data_path = "gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs.csv"
        artifact_path = "models/"
        os.makedirs(artifact_path, exist_ok=True)

        # 3. Auth GCP (si via secrets.toml → ajuster selon ton contexte)
        if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") is None:
            gcp_credentials_path = "./mlflow-trainer.json"  # ⚠️ À adapter
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_credentials_path
            if not os.path.exists(gcp_credentials_path):
                raise FileNotFoundError(f"Fichier de clé GCP introuvable : {gcp_credentials_path}")

    else:
        raise ValueError("Environnement invalide : choisir 'dev' ou 'prod'")
    
    return data_path, artifact_path


def load_and_clean_data(path: str):
    df = pd.read_csv(path, sep=";")
    df[['latitude', 'longitude']] = df['Coordonnées géographiques'].str.split(',', expand=True).astype(float)
    df_clean = df.dropna(subset=['latitude', 'longitude'])

    X = df_clean.drop(columns='Comptage horaire')
    y = df_clean['Comptage horaire']

    return X, y

import shutil

def train_rf(X, y, env, test_mode):
    y = y.to_numpy()
    run_name = f"RandomForest_Train_{env}" + ("_TEST" if test_mode else "")

    print("Tracking URI ACTIF :", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        print(f"🆔 MLflow Run ID : {run.info.run_id}")
        print(f"🆔 MLflow Run Name : {run_name}")

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

        temp_model_path = "tmp_rf_model"
        os.makedirs(temp_model_path, exist_ok=True)

        rf.save(os.path.join(temp_model_path, "rf"))
        mlflow.log_artifacts(temp_model_path, artifact_path="rf_model")

        print(f"📦 Modèle sauvegardé dans : {temp_model_path}")
        print(f"📤 Artefacts MLflow loggés dans : rf_model")
        print(f"📁 Artefacts visibles dans : {os.path.join(mlflow.get_tracking_uri().replace('file:', ''), run.info.experiment_id, run.info.run_id)}")

        shutil.rmtree(temp_model_path)  # nettoyage
        print(f"🧹 Répertoire temporaire supprimé : {temp_model_path}")



def train_nn(X, y, env, test_mode):
    y = y.to_numpy(dtype="float32")
    run_name = f"NeuralNet_Train_{env}" + ("_TEST" if test_mode else "")

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        print(f"🆔 MLflow Run ID : {run.info.run_id}")
        print(f"🆔 MLflow Run Name : {run_name}")

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

        temp_model_path = "tmp_nn_model"
        os.makedirs(temp_model_path, exist_ok=True)
        
        nn.save(os.path.join(temp_model_path, "nn"))
        mlflow.log_artifacts(temp_model_path, artifact_path="nn_model")

        print(f"📦 Modèle sauvegardé dans : {temp_model_path}")
        print(f"📤 Artefacts MLflow loggés dans : nn_model")
        print(f"📁 Artefacts visibles dans : {os.path.join(mlflow.get_tracking_uri().replace('file:', ''), run.info.experiment_id, run.info.run_id)}")

        shutil.rmtree(temp_model_path)  # nettoyage
        print(f"🧹 Répertoire temporaire supprimé : {temp_model_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train bike count models (RF + NN)")
    parser.add_argument('--model_test', action='store_true', help="Use 1000 samples for fast training")
    parser.add_argument('--env', default="dev", choices=["dev", "prod"], help="Select environment: dev or prod")
    args = parser.parse_args()

    # data_path = 1000 if args.model_test else None
    data_path, artifact_path = setup_environment(args.env, args.model_test)

    print(f"✅ Environnement {args.env} configuré : MLflow et artefacts prêts")
    print(f"✅ Chargement des données depuis {data_path}...")

    # import pandas as pd
    # df = pd.read_csv("gs://df_traffic_cyclist1/raw_data/comptage-velo-donnees-compteurs_test.csv", sep=";")
    # print(df.head())

    X, y = load_and_clean_data(data_path)
    print(f"✅ Données chargées : {X.shape[0]} échantillons pour l'environnement {args.env}")

    train_rf(X, y, env=args.env, test_mode=args.model_test)
    train_nn(X, y, env=args.env, test_mode=args.model_test)

    print(f"✅ Modèles entraînés et loggés avec MLflow ({args.env})")
