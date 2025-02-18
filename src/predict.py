def predict_risk(model, input_data):
    """Predict CVD risk for new data."""
    return model.predict_proba(input_data)[:, 1]