import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump, load


def saveModel(file, model):
    dump(model, file, compress=('gzip', 6))


def loadModel(file):
    try:
        return load(file)
    except FileNotFoundError:
        return None


# Creates new model
def newModel():
    return LinearRegression()


# Trains Model on given data
def train(df, model):
    # Create a new DataFrame with the necessary columns

    data = pd.DataFrame({
        'Previous_Close': df['Close'].shift(1),
        'Open': df['Open'],
        'Close': df['Close']
    })

    # Drop any rows with missing values
    data = data.dropna()
    # Separate the features and target variable
    X = data[['Previous_Close', 'Open']]  # Features
    y = data['Close']  # Target variable

    # Train the model on the entire dataset
    model.fit(X, y)

    return model


# Returns the predicted value and the date for the value
def process(df, model):
    # Get the previous day's close and current day's opening value
    prev_close = df['Close'].iloc[-1]
    current_open = df['Open'].iloc[-1]

    current_date = df.index[-1].strftime('%Y-%m-%d')  # Get date

    # If the current day's opening value is missing, use the last available opening value
    if pd.isnull(current_open):
        current_open = df['Open'].iloc[-2]

    # Create a DataFrame with the necessary columns and matching column names
    data = pd.DataFrame({
        'Previous_Close': [prev_close],
        'Open': [current_open]
    })

    # Predict the day's closing value using the model
    prediction = model.predict(data)

    # Get result
    return prediction[0], current_date
