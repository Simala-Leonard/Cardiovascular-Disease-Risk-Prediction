from src.data_preprocessing import load_data, preprocess_data
from src.train_model import train_model, evaluate_model

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data("data/framingham.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train and evaluate the model
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")