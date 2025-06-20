import streamlit as st

def show():

    st.header("1. Optimisation de la taille des images et des modèles")
    with st.expander("Détails"):
        st.write("""
        Pour rester dans les quotas du free tier (Render, Cloud Run, Cloud Functions), il faut absolument :
        - Compresser les modèles
        - Limiter les dépendances
        
        **Points clés :**
        - La taille des containers est cruciale : 
            - Render limite à 512MB
            - Cloud Functions à 2GB
        - Meilleure architecture trouvée : déploiement via Cloud Run avec 4Gi de mémoire
        - Répartition entre deux backends distincts :
            - regmodel-api
            - classmodel-api
        """)
        st.warning("Un seul backend pour tout ? Trop lourd. Deux microservices = déploiements plus efficaces.")

    st.header("2. Architecture de chargement des modèles")
    with st.expander("Évolution de l'approche"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Approche initiale")
            st.error("Problèmes :")
            st.write("- Modèles chargés depuis Streamlit Cloud")
            st.write("- Résultat : trop lent, fragile, crash")
        
        with col2:
            st.subheader("Deuxième approche")
            st.error("Problèmes :")
            st.write("- Chargement à chaque prédiction via API")
            st.write("- Trop coûteux avec plusieurs modèles")
        
        with col3:
            st.subheader("Solution optimale")
            st.success("Avantages :")
            st.write("- Chargement des 'best models' au démarrage")
            st.write("- Actualisation via endpoint /refresh_model")
            st.write("- Stockage GCS centralisé")

    st.header("3. Développement et DevOps")
    with st.expander("Bonnes pratiques"):
        st.write("""
        **Aller-retour dev/prod essentiel :**
        - Reproduire le comportement localement avec docker compose, curl et FastAPI
        - Tester la logique en local pour éviter les frustrations
        
        **Problématiques :**
        - Temps de build/déploiement longs sur Cloud Run/Render
        - Requêtes curl avec JSON de test indispensables pour itérer vite
        """)
        
        st.code("""
        # Exemple de test curl local
        curl -X POST "http://localhost:8000/predict" \\
        -H "Content-Type: application/json" \\
        -d '{"data": [...]}'
        """, language="bash")

    st.header("4. Gestion des environnements (DEV / PROD)")
    with st.expander("Comparaison des approches"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Avec Render")
            st.error("Limitations :")
            st.write("- Gestion d'environnements compliquée")
            st.write("- Peu flexible (Python, dépendances, secrets...)")
        
        with col2:
            st.subheader("Avec Docker")
            st.success("Avantages :")
            st.write("- Environnement sous contrôle")
            st.write("- Bugs apparaissent d'abord en local")
            st.write("- Reproductible et stable")
            st.write("- Maîtrise complète des versions et configurations")

    st.markdown("---")
    st.success("""
    **Conclusion générale :**  
    L'approche microservices avec des containers Docker bien optimisés et un cycle de développement local 
    rigoureux permet de construire des applications stables et efficaces malgré les limitations des free tiers.
    """)
