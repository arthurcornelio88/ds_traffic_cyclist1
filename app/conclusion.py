import streamlit as st
from streamlit_extras.let_it_rain import rain

def show():
    
    # Animation
    rain(
        emoji="✨",
        font_size=30,
        falling_speed=3,
        animation_length="infinite",
    )
    
    # Message principal
    st.markdown("""
    <div style='text-align: center; margin: 1rem;'>
        <h1 style='color: green;'>Merci pour votre attention !</h1>
        <p style='font-size: 1.2rem;'>
            Ce projet nous a appris énormément, et nous espérons que ce retour d'expérience<br>
            pourra être utile à d'autres passionnés de data et IA.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # GIF animé centré
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("https://media.giphy.com/media/3o7abKhOpu0NwenH3O/giphy.gif", width=300)
    
    # Citation
    st.markdown("""
                   <div style='text-align: center; margin: 1rem;'>
    <blockquote style='border-left: 3px solid #ccc; padding-left: 1rem; font-style: italic;'>
        " L’important ce n’est pas la destination, c’est le voyage !" 
                Robert Louis Stevenson
    </blockquote>
                  </div>
    """, unsafe_allow_html=True)