## 📦 Configurer Git LFS pour les modèles
```hcl
git lfs install
git lfs track "*.joblib"
git lfs track "*.keras"
git add .gitattributes
git add models/
git commit -m "Ajout des modèles avec Git LFS"
```
> ⚠️ Assure-toi que le quota LFS de ton compte GitHub est suffisant (~1 Go gratuit).

## Train
```hcl
python ./app.train.py --env dev              # Entraîne en local
python ./app.train.py --env prod             # Entraîne avec GCP/ngrok
python ./app.train.py --env dev --model_test # Train rapide local
```

### Lancer MLflow en local
```
mkdir -p mlruns/artifacts

# en prod
export GOOGLE_APPLICATION_CREDENTIALS=./mlflow-ui-access.json

mlflow server \
  --backend-store-uri file:./mlruns \
  --default-artifact-root gs://df_traffic_cyclist1/mlruns \
  --host 127.0.0.1 \
  --port 5000
  ```

### TODO
> Credentials GCP pour streamlit
```hcl
Tu peux créer un compte de service GCP avec les rôles :

Storage Object Viewer

AI Platform Viewer (si tu vas vers Vertex AI plus tard)

Puis :

Crée un fichier JSON de clé

Dans Streamlit Cloud, ajoute ce fichier dans secrets

Clé : gcp_service_account

Valeur : contenu du JSON
```
to-push
> bucket gcp + mlflow server --default-artifact-root gs://my-bucket/models 

Avec terraform ! gestion de credentials tb ! automatique.
  