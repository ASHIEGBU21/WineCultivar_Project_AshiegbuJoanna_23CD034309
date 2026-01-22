# model_development.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # ---------------------------------------------------
    # 1. Load the dataset (from provided file)
    # ---------------------------------------------------
    column_names = [
        'cultivar',
        'alcohol',
        'malic_acid',
        'ash',
        'alcalinity_of_ash',
        'magnesium',
        'total_phenols',
        'flavanoids',
        'nonflavanoid_phenols',
        'proanthocyanins',
        'color_intensity',
        'hue',
        'od280/od315_of_diluted_wines',
        'proline'
    ]

    data = pd.read_csv(
        "wine.data.txt",
        header=None,
        names=column_names
    )

    # ---------------------------------------------------
    # 2. Data Preprocessing
    # ---------------------------------------------------

    # 2.1 Handle missing values (if any)
    if data.isnull().sum().any():
        data.fillna(data.mean(), inplace=True)

    # 2.2 Feature selection (exactly six allowed features)
    selected_features = [
        'alcohol',
        'malic_acid',
        'alcalinity_of_ash',
        'total_phenols',
        'flavanoids',
        'proline'
    ]

    X = data[selected_features]
    y = data['cultivar']  # target variable

    # ---------------------------------------------------
    # 3. Train-test split
    # ---------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # ---------------------------------------------------
    # 4. Feature Scaling (MANDATORY)
    # ---------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------
    # 5. Model Implementation
    # ---------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # ---------------------------------------------------
    # 6. Model Training
    # ---------------------------------------------------
    model.fit(X_train_scaled, y_train)

    # ---------------------------------------------------
    # 7. Model Evaluation
    # ---------------------------------------------------
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # ---------------------------------------------------
    # 8. Save the trained model and scaler
    # ---------------------------------------------------
    joblib.dump(model, "wine_cultivar_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("\nModel and scaler saved successfully.")


if __name__ == "__main__":
    main()
