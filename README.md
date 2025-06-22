# Diabetic Macular Edema Detection (Streamlit + CNNs)

This project builds a web-based diagnostic tool to detect **Diabetic Macular Edema (DME)** from retinal images using deep learning. Three models are compared:

* ✅ Baseline CNN
* 🧱 Deep CNN
* 📱 Transfer Learning with MobileNetV2

It includes:

* 📊 Classification report + confusion matrix
* 🖼️ Image-based prediction with Streamlit UI
* 📁 Organized structure for training, deployment, and experimentation

---
## 🚀 Demo

This app is deployed via **Streamlit Cloud**.

🔗 **Live Demo**: [DME Detection Streamlit App](https://dmeproject0.streamlit.app/)



---
## 📁 Project Structure

```
DME_Detection_Project/
├── models/                 # Saved model weights (.h5)
│   ├── dme_model.h5
│   ├── dme_model_deep.h5
│   └── dme_model_mobilenet.h5
│
├── reports/               # Visual outputs
│   ├── accuracy_plot.png
│   ├── accuracy_plot_deep.png
│   ├── accuracy_plot_mobilenet.png
│   └── confusion_matrix.png
│
├── train/                 # Image dataset (flat + labels.csv)
│
├── labels.csv             # CSV file mapping filename to label
│
├── cnn.py                 # Baseline CNN
├── cnn_deep.py            # Deep CNN
├── cnn_mobilenet.py       # MobileNetV2 transfer learning
│
├── utils.py               # Image preprocessing & model helpers
├── streamlit_app.py       # Streamlit dashboard
├── requirements.txt       # All dependencies
└── README.md              # 📄 This file
```

---

## ⚙️ Setup

1. **Clone the repo**

```bash
git clone <your-repo-url>
cd DME_Detection_Project
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

4. **Train models** (Optional if .h5 files are available)

```bash
python cnn.py
python cnn_deep.py
python cnn_mobilenet.py
```

5. **Launch Streamlit app**

```bash
streamlit run streamlit_app.py
```

---

## 🧠 Models

### cnn.py

* Simple CNN with 3 conv layers

### cnn\_deep.py

* 4 conv layers with 256 units + dropout + batch norm

### cnn\_mobilenet.py

* Fine-tuned MobileNetV2 using transfer learning

---

## 📈 Evaluation Metrics

* Accuracy
* Classification Report
* Confusion Matrix

These are saved in `/reports`.

---

## 🔮 Prediction Labels

```
['Mild', 'Moderate', 'Normal', 'Proliferate', 'Severe']
```

---

## 📦 Requirements (partial)

* streamlit
* tensorflow
* scikit-learn
* pandas
* opencv-python
* matplotlib
* seaborn

---



## 👤 Author

* Built by Vachana Visweswaraiah as part of a DME classification project using deep learning and Streamlit.
