import streamlit as st
import pandas as pd

def show():
    st.header("🤖 Retour d'Expérience - Modélisation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Succès")
        st.success("""
        - Choix des modèles et entrainement
        - Optimisation des hyperparamètres efficace
        - Pipeline reproductible (détails déploiement)
        """)
    
    with col2:
        st.subheader("Défis")
        st.error("""
        - Temps d'entrainement long (modèle de classification et de deeplearning - lié au volume de données traitées)
        """)
    
    st.subheader("Pistes d'amélioration")
    st.error("""
        - Ajouter des variables contextuelles (météo, événements)
        - Tester et mésurer les performances d'autres modèles de ML sur notre problème.
    """)
    