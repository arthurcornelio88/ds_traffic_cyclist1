import streamlit as st

def show():
    st.header("🔍 Retour d'Expérience - Analyse Exploratoire")
    
    st.subheader("Ce qui a bien fonctionné")
    st.success("""
    - Récupération et formatage adéquat des données
    - Visualisations interactives des données avec Plotly sous jupyter / Google collab
    """)
    
    st.subheader("Difficultés rencontrées")
    st.error("""
    - intégration des visualisations interactives Plotly sous Streamlit
    """)
    