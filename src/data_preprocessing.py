import pandas as pd # type: ignore

# URL of the raw CSV file on GitHub
url = "https://github.com/Simala-Leonard/Datasets/raw/refs/heads/main/framingham.csv"


# Load the data into a DataFrame
df = pd.read_csv(url)

def preprocess_data(df):
    """Preprocess the data."""
    df = df.dropna()  # Handle missing values
    X = df.drop("TenYearCHD", axis=1)  # Features
    y = df["TenYearCHD"]  # Target
    return train_test_split(X, y, test_size=0.2, random_state=42) # type: ignore