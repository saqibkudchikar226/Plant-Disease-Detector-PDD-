import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# =========================
# PAGE
# =========================

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿"
)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "model/plant_model.keras"
)

# =========================
# LOAD REAL CLASS NAMES
# =========================

with open("class_names.json","r") as f:
    class_names = json.load(f)

# =========================
# FERTILIZER
# =========================

fertilizer_dict = {

    "Tomato Early blight leaf":
    "Copper Fungicide",

    "Tomato leaf late blight":
    "Mancozeb Spray",

    "Potato leaf early blight":
    "Copper Fungicide",

    "Apple rust leaf":
    "Sulfur Fungicide"
}

# =========================
# TITLE
# =========================

st.title("🌿 Plant Disease Detector")

uploaded_file = st.file_uploader(
    "Upload Plant Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,width=300)

    # =========================
    # PREPROCESS
    # =========================

    img = image.resize((224,224))

    img_array = np.array(img)/255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # =========================
    # PREDICT
    # =========================

    prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)

    confidence = np.max(prediction)*100

    result = class_names[predicted_index]

    # =========================
    # PLANT NAME
    # =========================

    plant = result.split()[0]

    # =========================
    # HEALTH STATUS
    # =========================

    if "spot" in result.lower() \
    or "blight" in result.lower() \
    or "virus" in result.lower() \
    or "rust" in result.lower() \
    or "mold" in result.lower() \
    or "mildew" in result.lower():

        health = "Diseased ❌"

    else:
        health = "Healthy ✅"

    # =========================
    # FERTILIZER
    # =========================

    fertilizer = fertilizer_dict.get(
        result,
        "Organic Compost Recommended"
    )

    # =========================
    # OUTPUT
    # =========================


    st.success(f"🌱 Plant: {plant}")

    st.error(f"🦠 Disease: {result}")

    st.info(f"💚 Health Status: {health}")

    st.warning(
        f"📊 Confidence: {confidence:.2f}%"
    )

    st.success(
        f"🧪 Recommended Fertilizer: {fertilizer}"
    )