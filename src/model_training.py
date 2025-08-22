import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

def clean_outliers_per_group(
    df: pd.DataFrame, group_col: str, num_cols: list[str]
) -> pd.DataFrame:
    def filter_group(group: pd.DataFrame) -> pd.DataFrame:
        for col in num_cols:
            q1 = group[col].quantile(0.25)
            q3 = group[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            group = group[(group[col] >= lower) & (group[col] <= upper)]
        return group

    return df.groupby(group_col, group_keys=False).apply(filter_group)

def train_model() -> RandomizedSearchCV:
    df = load_data()

    features = ["traffic_levels", "weather_conditions", "sequence_in_delivery", "is_weekend"]

    categorical = ["traffic_levels", "weather_conditions"]
    numerical = ["sequence_in_delivery", "is_weekend"]
    df_clean = clean_outliers_per_group(df, "delivery_time_window", numerical)

    X = df_clean[features]
    y = df_clean["delivery_time_window"]

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

    random_search = RandomizedSearchCV(
        pipe, param_grid, cv=5, scoring="f1_macro", n_jobs=-1, verbose=5, n_iter=10
    )
    random_search.fit(X_train, y_train)

    print("Best Parameters:", random_search.best_params_)
    y_pred = random_search.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return random_search


if __name__ == "__main__":
    model = train_model()
    if not os.path.isdir('models'):
        os.makedirs('models')
    joblib.dump(model, 'models/parcel_delivery_model.pkl')