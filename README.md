# 🚢 Titanic Survival Prediction: Machine Learning Pipeline

This project is a complete machine learning pipeline designed to predict which passengers survived the sinking of the RMS Titanic. It focuses on data cleaning, preparation, and comparing the performance of three classic classification models.

## 1. Project Goal

The main goal is to build models that can accurately predict the target variable, **`Survived`** (where `1` means the passenger survived, and `0` means they did not), based on their personal and travel details.

## 2. Dataset and Preprocessing

The script uses the standard Titanic training dataset (`train.csv`). Before modeling, the data goes through several cleaning steps:

### Data Cleaning Highlights

| Feature | Missing Value Strategy | Transformation |
| :--- | :--- | :--- |
| **Age** | Filled with the **Median** age. | Scaled (StandardScaler). |
| **Embarked** | Filled with the most common value (**Mode**). | One-Hot Encoded. |
| **Cabin** | Filled with **'Unknown'**. | One-Hot Encoded. |
| **Fare** | N/A | Scaled (StandardScaler). |
| **Sex** and **Pclass** | N/A | One-Hot Encoded. |

### Features Dropped

The following identifying columns were removed because they don't help the model predict survival: `PassengerId`, `Name`, and `Ticket`.

## 3. Machine Learning Models Compared

The dataset is split into 80% for training and 20% for testing. We compare the results of three popular classification algorithms:

1.  **Logistic Regression:** A basic, linear model to establish a baseline performance.
2.  **Random Forest Classifier:** A powerful ensemble model known for high accuracy.
3.  **Decision Tree Classifier:** A model that creates simple, understandable rules for classification.

### Evaluation

Each model is evaluated using:
* **Test Accuracy:** How well the model performs on unseen data.
* **5-Fold Cross-Validation Score:** A measure of how stable and reliable the model is across different subsets of the data.
* **Classification Report:** Detailed metrics including Precision, Recall, and F1-Score.

## 4. How to Run the Script

To get started with this project, follow these steps:

1.  **Dependencies:** You need to install the following Python libraries:
    ```bash
    pip install pandas scikit-learn
    ```

2.  **Data Setup:** Ensure you have the `train.csv` file placed inside a directory named **`./Dataset/`** in the same folder as your Python script.

3.  **Execution:** Run the main script from your terminal:
    ```bash
    python your_script_name.py
    ```
