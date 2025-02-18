import pytest # type: ignore
from src.train_model import train_model
from sklearn.datasets import make_classification # type: ignore

def test_model_training():
    """Test if the model trains correctly."""
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    model = train_model(X, y)
    assert model is not None