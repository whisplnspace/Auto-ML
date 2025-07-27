# 🤖 AutoML Trainer with Smart Algorithm Selection

This Streamlit web application allows users to upload a CSV dataset and automatically trains the best machine learning model based on data analysis. It intelligently detects whether the task is classification or regression, preprocesses the data, evaluates multiple models, and provides performance metrics along with a downloadable trained model.

## 🚀 Features

- 🧠 **Automatic Task Detection** (Classification or Regression)
- 📊 **Data Preprocessing** with missing value handling, scaling, and encoding
- 🏆 **Model Evaluation** and **Best Model Selection**
- 📈 Displays **Classification Report**, **Confusion Matrix**, and **Regression Metrics**
- 💾 **Download Trained Model** in `.pkl` format
- 🖥️ Clean, intuitive **Streamlit UI**

## 🖼️ Demo Preview

![screenshot](https://user-images.githubusercontent.com/your-screenshot-link.png)

## 📂 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/automl-trainer.git
cd automl-trainer
````

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the App

```bash
streamlit run app.py
```

> Make sure you have your dataset in `.csv` format ready for upload in the app.

## 📦 Requirements

* `streamlit`
* `pandas`
* `numpy`
* `scikit-learn`
* `joblib`

Install them with:

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

## 🧠 Models Used

* **Classification**

  * Logistic Regression
  * Random Forest Classifier
  * Support Vector Classifier (SVC)

* **Regression**

  * Linear Regression
  * Random Forest Regressor
  * Support Vector Regressor (SVR)

## 📁 Folder Structure

```
automl-trainer/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── saved_models/         # Folder to store trained models
└── README.md             # Project documentation
```

## 📤 Example Output

Once a model is trained, you’ll see:

* Best model name and score
* Evaluation metrics
* Option to download the model as a `.pkl` file

## ✅ Use Cases

* Rapid prototyping and testing of datasets
* Educational demonstrations of ML model selection
* Quick evaluations for data science projects

## 📌 Future Improvements

* Add support for multi-class classification
* Integrate hyperparameter tuning (e.g., GridSearchCV)
* Visualize feature importance and model performance curves
* Add support for saving metadata and config

## 📃 License

MIT License

---

Made with ❤️ 

```

