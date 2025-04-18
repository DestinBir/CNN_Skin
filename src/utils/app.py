import streamlit as st
from PIL import Image
import numpy as np
#from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import img_to_array

# Chargement du modèle
# model = load_model('mon_model_cnn_skin.h5')  # Assure-toi que ce chemin est correct

# Dictionnaire des classes
class_names = ['Acné', 'Vitiligo', 'Hyperpigmentation', 'Nail Psoriasis', 'SJS-TEN']

# Interface Streamlit
st.set_page_config(page_title="CNN Médical en Temps Réel", layout="centered")
st.title("🧠 Classification d’Images Médicales avec CNN")
st.markdown("---")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Health_Medical_icon.png", width=100)
st.sidebar.title("Navigation")
section = st.sidebar.radio("Aller à", ["🏁 Présentation", "📊 Dataset", "🔍 Tester une Image"])

# Section présentation
if section == "🏁 Présentation":
    st.header("Pourquoi ce projet ?")
    st.write("""
    Les maladies de la peau représentent une **préoccupation majeure en Afrique**, souvent négligée.
    Ce projet démontre comment les **réseaux de neurones convolutifs (CNN)** peuvent aider à **classifier des images dermatologiques** en temps réel.

    > _“L’IA ne remplacera pas les médecins, mais les médecins qui l’utilisent remplaceront ceux qui ne le font pas.”_
    """)
    st.markdown("#### Avantages :")
    st.markdown("- 📸 Traitement d’images en temps réel")
    st.markdown("- 🧬 Détection de maladies rares")
    st.markdown("- ⚙️ Utilisation de Transfer Learning (ResNet)")

# Section dataset
elif section == "📊 Dataset":
    st.header("À propos du Dataset")
    st.write("""
    Le dataset est composé de **9548 images dermatoscopiques** réparties en 5 classes :
    """)
    data = {
        "Acné": 1148,
        "Vitiligo": 2016,
        "Hyperpigmentation": 700,
        "Nail Psoriasis": 2520,
        "SJS-TEN": 3164
    }
    st.bar_chart(data)

    st.info("Les données proviennent de différents hôpitaux et ressources en ligne, permettant une diversité d’origines géographiques.")

# Section prédiction
elif section == "🔍 Tester une Image":
    st.header("Tester une Image de Peau")
    uploaded_file = st.file_uploader("Upload une image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).resize((224, 224))
        st.image(image, caption="Image uploadée", use_column_width=True)

        # Prétraitement
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # Prédiction
        prediction = model.predict(image_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"### ✅ Diagnostic possible : **{predicted_class}**")
        st.markdown("> Cette prédiction est à usage **éducatif** uniquement.")

