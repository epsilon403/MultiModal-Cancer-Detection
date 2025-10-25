import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np


st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("🧠 Détection de tumeurs cérébrales avec YOLOv8")
st.markdown("Téléversez une image IRM pour détecter le type de tumeur.")


st.sidebar.header("⚙️ Choix du modèle")

model_type = st.sidebar.radio(
    "Sélectionnez le modèle à utiliser :",
    ("Modèle sans augmentation", "Modèle avec augmentation")
)

if model_type == "Modèle sans augmentation":
    default_model_path = "runs/detect/train/weights/best.pt"  
else:
    default_model_path = "runs/detect/train2/weights/best.pt"  

model_path = st.sidebar.text_input("Chemin du modèle YOLO :", default_model_path)

try:
    model = YOLO(model_path)
    st.sidebar.success(f"✅ Modèle chargé : {model_path}")
except Exception as e:
    st.sidebar.error(f"Erreur lors du chargement du modèle : {e}")
    model = None


uploaded_file = st.file_uploader("📤 Téléversez une image (IRM)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🩺 Image originale", use_column_width=True)

    if st.button("🔍 Lancer la détection"):
        if model is None:
            st.error("Veuillez charger un modèle YOLO valide avant de lancer la détection.")
        else:
            st.info("Prédiction en cours...")
            
            results = model.predict(image, imgsz=640, conf=0.25)

            result_img = results[0].plot()  
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            names = model.names
            detected_classes = [names[int(box.cls)] for box in results[0].boxes]
            st.success(f"🧠 Tumeur détectée : {', '.join(set(detected_classes)) if detected_classes else 'Aucune tumeur détectée.'}")

            st.image(result_img, caption="Résultat de la détection", use_column_width=True)
