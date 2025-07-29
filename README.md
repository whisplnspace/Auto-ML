<h1 align="center">🤖 AutoML Trainer with Smart Algorithm Selection</h1>

<p align="center">
  A powerful, no-code Streamlit app that auto-detects your dataset type, trains the best ML model, evaluates performance, and lets you download the trained model.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Scikit--Learn-AutoML-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.8+-yellow?style=flat-square" />
</p>

---

## 📸 App Preview

<p align="center">
  <img src="https://your-image-link.app-preview.gif" width="700" alt="App Demo Preview"/>
</p>

---

## 🧭 Workflow Diagram

```mermaid
graph TD
    A[Upload CSV Dataset] --> B[Detect Task Type]
    B --> C[Preprocess Data]
    C --> D[Train ML Models]
    D --> E[Select Best Model]
    E --> F[Display Metrics]
    F --> G[Download Model]

````

---

## ✨ Features

* 🧠 **Task Detection**: Automatically distinguishes between classification and regression.
* 🧹 **Smart Preprocessing**: Handles missing data, encodes categories, and scales numerics.
* 🏁 **Model Benchmarking**: Trains and compares multiple models.
* 📈 **Performance Report**: Confusion matrix, classification report or regression metrics.
* 💾 **Model Export**: Download trained `.pkl` model file instantly.

---

## 🚀 Getting Started

### 🔧 1. Clone this repository

```bash
git clone https://github.com/whisplnspace/automl-trainer.git
cd automl-trainer
```

### 🧪 2. Create & Activate Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate        # For Linux/macOS
venv\Scripts\activate           # For Windows
```

### 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ 4. Launch the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
automl-trainer/
│
├── app.py                # 🚀 Main Streamlit App
├── saved_models/         # 💾 Stores Exported Models
├── requirements.txt      # 📦 Required Python Packages
└── README.md             # 📘 Project Documentation
```

---

## 📊 Supported Algorithms

### 🔍 Classification

* Logistic Regression
* Random Forest Classifier
* Support Vector Classifier (SVC)

### 📈 Regression

* Linear Regression
* Random Forest Regressor
* Support Vector Regressor (SVR)

---

## 📌 Use Cases

* Rapid model benchmarking without writing code
* Educational demonstrations of ML pipeline
* Exploratory model analysis for datasets
* Building production-ready `.pkl` models quickly

---

## 📥 Sample Output

<p align="center">
  <img src="https://your-image-link.confusion-matrix.png" width="600" alt="Confusion Matrix Output"/>
</p>

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome!
If you’d like to improve this app, feel free to fork and submit a pull request.

---

## 💡 Built With

* [Streamlit](https://streamlit.io/)
* [Scikit-Learn](https://scikit-learn.org/)
* [Pandas](https://pandas.pydata.org/)
* [Joblib](https://joblib.readthedocs.io/)

---

Made with ❤️ 


