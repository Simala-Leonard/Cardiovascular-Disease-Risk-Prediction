from xgboost import XGBClassifier # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

def train_model(X_train, y_train):
    """Train an XGBoost model."""
    model = XGBClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)