

# Import necessary libraries
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set random seed for reproducibility
RANDOM_SEED = 42


def load_data(file_path):
    """Load the Titanic dataset from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess the dataset by handling missing values."""
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Fill missing 'Embarked' with mode
    df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])
    
    # Fill missing 'Age' with median
    df_processed['Age'] = df_processed['Age'].fillna(df_processed['Age'].median())
    
    # Fill missing 'Cabin' with 'Unknown'
    df_processed['Cabin'] = df_processed['Cabin'].fillna('Unknown')
    
    print("Missing values handled:")
    print(df_processed.isnull().sum())
    return df_processed


def prepare_features(df):
    """Prepare features and target for modeling."""
    # Define features and target
    X = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
    y = df['Survived']
    
    # Encode categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare']
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    return X, y, scaler

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple machine learning models."""
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_SEED),
        'Decision Tree': DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED)
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        # Print results
        print(f'\n{name} Results:')
        print(f'Cross-Validation Accuracy: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})')
        print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
        print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

def main():
    """Main function to execute the Titanic survival prediction pipeline."""
    print("Starting Titanic Survival Prediction Pipeline...")
    
    # Load data
    data = load_data('./Dataset/train.csv')
    if data is None:
        return
    
    # Display dataset info
    print("\nDataset Info:")
    print(data.info())
    
    # Preprocess data
    data_processed = preprocess_data(data)
    
    # Prepare features
    X, y, scaler = prepare_features(data_processed)
    
    # Train and evaluate models
    train_and_evaluate_models(X, y)
    
    print("\nPipeline completed. Check 'survival_by_embarked.png' for EDA visualization.")

if __name__ == "__main__":
    main()