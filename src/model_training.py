import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/parcel_delivery_dataset.csv")
    df.dropna(inplace=True)
    df["dates"] = pd.to_datetime(df["dates"])

    df["day_of_week"] = df["dates"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df

def train_model() -> GridSearchCV:
    df = load_data()

    features = ["traffic_levels", "weather_conditions", "sequence_in_delivery", "is_weekend"]
    X = df[features]
    y = df["delivery_time_window"]

    categorical = ["traffic_levels", "weather_conditions", "parcel_size"]
    numerical = ["sequence_in_delivery", "is_weekend", "distance", "parcel_weights"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numerical),
        ]
    )

    rf = RandomForestClassifier(random_state=42)

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", rf),
    ])

    param_grid = [
        {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__min_samples_split": [2, 4],
        }
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    grid_search = GridSearchCV(
        pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=5
    )
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    y_pred = grid_search.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return grid_search


if __name__ == "__main__":
    model = train_model()
    if not os.path.isdir('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/parcel_delivery_model.pkl')