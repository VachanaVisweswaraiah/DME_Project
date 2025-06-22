import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Diabetic Retinopathy Detector", layout="wide")

MODEL_PATHS = {
    "CNN": "models/dme_model.h5",
    "Deep CNN": "models/dme_model_deep.h5",
    "MobileNetV2": "models/dme_model_mobilenet.h5"
}

IMG_SIZE = 150
CLASS_NAMES = ['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']

SAMPLE_IMAGES = {
    "Normal": "sample_images/sample_normal.png",
    "Mild": "sample_images/sample_mild.png",
    "Moderate": "sample_images/sample_moderate.png",
    "Severe": "sample_images/sample_severe.png",
    "Proliferate": "sample_images/sample_proliferate.png"
}


@st.cache_resource
def load_models():
    return {name: load_model(path) for name, path in MODEL_PATHS.items()}


models = load_models()

# ------------------- SIDEBAR -------------------
st.sidebar.title("DME Analyzer")
selected_tab = st.sidebar.radio(
    "Navigation",
    ["📖 Project Info", "🧠 Predict DR Stage", "📊 Model Evaluation"]
)


# ------------------- INFO PAGE -------------------
if selected_tab == "📖 Project Info":
    st.title("📘 Project Overview: Diabetic Retinopathy Detection")
    st.markdown("""
    This application helps detect **Diabetic Retinopathy (DR)** from retina images using three deep learning models:

    - **CNN** – Simple convolutional model.
    - **Deep CNN** – Deeper CNN with more layers.
    - **MobileNetV2** – Lightweight, fine-tuned MobileNetV2 model.

    ### DR Classification Stages:
    - **Normal**: No signs of DR.
    - **Mild**: Small microaneurysms.
    - **Moderate**: More noticeable changes, including blood vessels.
    - **Severe**: Extensive changes and blocked blood vessels.
    - **Proliferate**: Growth of new abnormal blood vessels.

    Upload or select a sample image to view predictions from all models.
    """)

# ------------------- PREDICTION PAGE -------------------
elif selected_tab == "🧠 Predict DR Stage":
    st.title("🧠 Diabetic Retinopathy Detection")
    st.markdown("Upload a retina image or try a sample to compare predictions from all models.")

    # ---------- 1. Uploaded Image ----------
    st.subheader("📤 Upload Retina Image")
    uploaded_file = st.file_uploader("Upload Retina Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Retina Image", width=300)

        # Preprocess
        img_array = np.array(image)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        tabs = st.tabs(["CNN", "Deep CNN", "MobileNetV2"])
        for tab, (model_name, model) in zip(tabs, models.items()):
            with tab:
                st.header(f"{model_name}")
                pred = model.predict(img_array)[0]
                label = CLASS_NAMES[np.argmax(pred)]
                conf = pred[np.argmax(pred)] * 100
                st.success(f"**Prediction:** {label} ({conf:.2f}%)")
                st.subheader("Class Probabilities")
                st.bar_chart({CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))})

    # ---------- 2. Sample Dropdown Image ----------
    with st.expander("📁 Try with a Sample Image Instead"):
        selected_sample = st.selectbox("Choose a sample retina image", list(SAMPLE_IMAGES.keys()))

        if selected_sample:
            path = SAMPLE_IMAGES[selected_sample]
            image = Image.open(path).convert('RGB')
            st.image(image, caption=f"Sample Image: {selected_sample}", width=300)

            # Preprocess
            img_array = np.array(image)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            tabs = st.tabs(["CNN (Sample)", "Deep CNN (Sample)", "MobileNetV2 (Sample)"])
            for tab, (model_name, model) in zip(tabs, models.items()):
                with tab:
                    st.header(f"{model_name}")
                    pred = model.predict(img_array)[0]
                    label = CLASS_NAMES[np.argmax(pred)]
                    conf = pred[np.argmax(pred)] * 100
                    st.success(f"**Prediction:** {label} ({conf:.2f}%)")
                    st.subheader("Class Probabilities")
                    st.bar_chart({CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))})


# ------------------- EVALUATION PAGE -------------------
elif selected_tab == "📊 Model Evaluation":
    st.title("📊 Model Evaluation")
    st.markdown("""
    Here's how each model performed on the validation dataset.
    
    - Metrics shown: **Accuracy**, **Confusion Matrix**
    - Data Source: Validation set (20% of training data)
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("CNN")
        st.image("reports/accuracy_plot.png", caption="Accuracy - CNN", use_column_width=True)
        st.image("reports/confusion_matrix.png", caption="Confusion Matrix - CNN", use_column_width=True)

    with col2:
        st.subheader("Deep CNN")
        st.image("reports/accuracy_plot_deep.png", caption="Accuracy - Deep CNN", use_column_width=True)
        st.image("reports/confusion_matrix_deep.png", caption="Confusion Matrix - Deep CNN", use_column_width=True)

    with col3:
        st.subheader("MobileNetV2")
        st.image("reports/accuracy_plot_mobilenet.png", caption="Accuracy - MobileNetV2", use_column_width=True)
        st.image("reports/confusion_matrix_mobilenet.png", caption="Confusion Matrix - MobileNetV2", use_column_width=True)

    st.markdown("---")
    st.markdown("Even advanced models like MobileNetV2 may misclassify borderline cases.\nExplore confusion matrices to see patterns in errors.")
