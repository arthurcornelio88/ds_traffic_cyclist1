import streamlit as st
import pandas as pd
from app.mlflow_model_loader import load_pipeline_from_mlflow
import os
import json

# === Initialisation GCP (si secrets présents) ===
if "gcp_service_account" in st.secrets:
    with open("/tmp/gcp.json", "w") as f:
        json.dump(st.secrets["gcp_service_account"], f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/gcp.json"

# === Chargement des meilleurs modèles via MLflow ===#
@st.cache_resource
def load_best_pipeline(model_type: str):
    return load_pipeline_from_mlflow(model_type, env="dev", test_mode=False)

rf_pipeline = load_best_pipeline("rf")
st.write("✅ Random Forest chargé :", type(rf_pipeline))

nn_pipeline = load_best_pipeline("nn")
st.write("✅ Neural Net chargé :", type(nn_pipeline))

# === UI ===
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choisissez une page :", ["🔍 Prédiction exemple", "📂 Prédiction CSV batch"])
st.title("🚲 Prédiction du comptage horaire de vélos")
modele = st.radio("Modèle à utiliser :", ["Random Forest", "Réseau de Neurones"])

# === Exemples manuels
raw_samples = [
    {
        'nom_du_compteur': 'Totem 73 boulevard de Sébastopol S-N',
        'date_et_heure_de_comptage': '2025-05-17 18:00:00+02:00',
        'coordonnées_géographiques': '48.8672, 2.3501',
        'mois_annee_comptage': 'mai 2025'
    },
    {
        'nom_du_compteur': '35 boulevard de Ménilmontant NO-SE',
        'date_et_heure_de_comptage': '2024-11-12 08:00:00+02:00',
        'coordonnées_géographiques': '48.8639, 2.3895',
        'mois_annee_comptage': 'novembre 2024'
    },
    {
        'nom_du_compteur': '102 boulevard de Magenta SE-NO',
        'date_et_heure_de_comptage': '2024-06-03 15:00:00+02:00',
        'coordonnées_géographiques': '48.8784, 2.3574',
        'mois_annee_comptage': 'juin 2024'
    }
]

# === Page 1 : prédiction manuelle
if page == "🔍 Prédiction exemple":
    idx = st.selectbox("Sélectionnez une observation brute à prédire :", range(len(raw_samples)), format_func=lambda i: f"Exemple {i+1}")
    sample_df = pd.DataFrame([raw_samples[idx]])

    try:
        if modele == "Random Forest":
            pred = rf_pipeline.predict_clean(sample_df)[0]
        else:
            pred = nn_pipeline.predict_clean(sample_df)[0][0]

        st.markdown("### 🔍 Observation sélectionnée")
        st.json(raw_samples[idx])
        st.success(f"🧾 Prédiction du comptage horaire : **{round(pred)} vélos**")

    except Exception as e:
        st.error("Erreur lors de la prédiction :")
        st.code(str(e))

# === Page 2 : prédiction sur CSV
elif page == "📂 Prédiction CSV batch":
    st.header("Prédiction sur fichier CSV brut")
    uploaded_file = st.file_uploader("Chargez un fichier brut (.csv)", type="csv")

    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        try:
            if modele == "Random Forest":
                predictions = rf_pipeline.predict(df_csv)
            else:
                predictions = nn_pipeline.predict(df_csv).flatten()

            df_csv['prediction_comptage_horaire'] = predictions.round().astype(int)

            st.markdown("✅ **Résultats :**")
            st.dataframe(df_csv.head(20))

            csv_output = df_csv.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger les résultats", csv_output, file_name="predictions_comptage.csv", mime="text/csv")

        except Exception as e:
            st.error("Erreur lors de la prédiction :")
            st.code(str(e))
