# Diabetic Macular Edema Detection

This project builds a web-based diagnostic tool to detect **Diabetic Macular Edema (DME)** from retinal images using deep learning. Three models are compared:

* ✅ Baseline CNN
* 🧱 Deep CNN
* 📱 Transfer Learning with MobileNetV2

It includes:

* 📊 Classification report + confusion matrix
* 🖼️ Image-based prediction with Streamlit UI
* 📁 Organized structure for training, deployment, and experimentation
* 🧪 Evaluation visualizations and pretrained models

---

## 📁 Project Structure

```
DME_Detection_Project/
├── app/
│   └── streamlit_app.py            # Streamlit UI dashboard
│
├── models/                         # Saved model weights (.h5)
│   ├── dme_model.h5
│   ├── dme_model_deep.h5
│   └── dme_model_mobilenet.h5
│
├── reports/                        # Output plots + confusion matrices
│   ├── accuracy_plot.png
│   ├── confusion_matrix.png
│   ├── accuracy_plot_deep.png
│   ├── confusion_matrix_deep.png
│   ├── accuracy_plot_mobilenet.png
│   └── confusion_matrix_mobilenet.png
│
├── sample_images/                  # Sample retina images for demo
│
├── train/                          # Image dataset (flat folder)
├── labels.csv                      # Maps filenames to labels
│
├── cnn.py                          # Baseline CNN training
├── cnn_deep.py                     # Deeper CNN architecture
├── cnn_mobilenet.py                # Transfer learning (MobileNetV2)
├── utils.py                        # Image pre-processing & helpers
├── requirements.txt                # Python dependencies
├── packages.txt                    # Linux dependencies (e.g., libgl1)
├── runtime.txt                     # Python version for Streamlit Cloud
└── README.md                       # 📄 This file
```

---

## 🚀 Demo

This app is deployed via **Streamlit Cloud**.

Try the live web app here 👉 [DME Detection Streamlit Demo](https://dmeproject0.streamlit.app/)

📦 App file: `app/streamlit_app.py`  
📁 Models auto-loaded from `models/`  
📊 Evaluation results read from `reports/`

---

## ⚙️ Setup

1. **Clone the repo**

```bash
git clone https://github.com/VachanaVisweswaraiah/DME_Project.git
cd DME_Project
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install requirements**

```bash
pip install -r requirements.txt
```

4. **Train models (optional)**

```bash
python cnn.py
python cnn_deep.py
python cnn_mobilenet.py
```

5. **Launch Streamlit app**

```bash
streamlit run app/streamlit_app.py
```

---

## 💻 Streamlit Dashboard (`streamlit_app.py`)

Tabs:

- **Project Info** – Overview & descriptions of all 3 models
- **Predict DR Stage** – Upload a retina image or select a sample to run predictions
- **Model Evaluation** – Visual display of confusion matrices and accuracy plots

---

## 🧠 Training Scripts

| Script            | Description                                    | Output files generated                          |
|-------------------|------------------------------------------------|-------------------------------------------------|
| `cnn.py`          | Baseline CNN with 3 conv layers               | `dme_model.h5`, accuracy/confusion plots        |
| `cnn_deep.py`     | Deeper CNN with batch norm + dropout         | `dme_model_deep.h5`, deep evaluation plots      |
| `cnn_mobilenet.py`| Transfer learning using MobileNetV2          | `dme_model_mobilenet.h5`, mobilenet eval plots  |

---

## 📈 Evaluation Metrics

Saved in `reports/` folder:

- Accuracy plots
- Confusion matrices
- Classification reports

---

## 🔮 Prediction Labels

```
['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']
```

---

## 📦 Dependencies

From `requirements.txt`:

- streamlit
- tensorflow==2.13.0
- scikit-learn
- pandas
- opencv-python
- matplotlib
- seaborn

From `packages.txt` (for Streamlit Cloud):

```
libgl1-mesa-glx
```

---

## 👤 Author

Built by [Vachana Visweswaraiah](https://github.com/VachanaVisweswaraiah) as part of a deep learning project on diabetic macular edema detection.

---

## 📌 Notes

- Ensure `runtime.txt` is set to `python-3.10` for compatibility on Streamlit Cloud
- Deployment may take a few minutes during the first launch
