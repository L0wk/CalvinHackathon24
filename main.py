# from requests_html import HTMLSession
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# from keras.optimizers import Adam
# from sklearn.metrics import mean_squared_error, r2_score


# # Function to fetch stock data from Yahoo Finance
# def fetch_stock_data(symbol):
#     url = f"https://finance.yahoo.com/quote/{symbol}/history?p={symbol}"
#     session = HTMLSession()
#     r = session.get(url)
#     rows = r.html.xpath("//table/tbody/tr")
#     data = []
#     for row in rows:
#         if len(row.xpath(".//td")) < 7:
#             continue
#         data.append(
#             {
#                 "Symbol": symbol,
#                 "Date": row.xpath(".//td[1]/span/text()")[0],
#                 "Open": row.xpath(".//td[2]/span/text()")[0],
#                 "High": row.xpath(".//td[3]/span/text()")[0],
#                 "Low": row.xpath(".//td[4]/span/text()")[0],
#                 "Close": row.xpath(".//td[5]/span/text()")[0],
#                 "Adj Close": row.xpath(".//td[6]/span/text()")[0],
#                 "Volume": row.xpath(".//td[7]/span/text()")[0],
#             }
#         )
#     df = pd.DataFrame(data)
#     df["Date"] = pd.to_datetime(df["Date"])
#     str_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
#     df[str_cols] = df[str_cols].replace(",", "", regex=True).astype(float)
#     df.dropna(inplace=True)
#     df = df.set_index("Date")
#     return df


# # Function to train LSTM model on stock data
# def train_lstm_model(X_train, y_train, X_test, y_test):
#     model = Sequential()
#     model.add(
#         LSTM(
#             32,
#             activation="relu",
#             return_sequences=True,
#             input_shape=(1, X_train.shape[2]),
#         )
#     )
#     model.add(Dropout(0.2))
#     model.add(LSTM(32, activation="relu", return_sequences=False))
#     model.add(Dense(1))
#     model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.004))
#     model.fit(
#         X_train,
#         y_train,
#         epochs=100,
#         batch_size=8,
#         validation_data=(X_test, y_test),
#         verbose=0,
#     )
#     return model


# # Function to plot actual vs predicted values
# def plot_predictions(y_test, y_pred, title):
#     plt.figure(figsize=(15, 6))
#     plt.plot(y_test, label="Actual Value", color="blue")
#     plt.plot(y_pred, label="Predicted Value", color="orange")
#     plt.ylabel("Adjusted Close (Scaled)")
#     plt.xlabel("Time Scale")
#     plt.legend()
#     plt.title(title)
#     plt.grid(True)
#     plt.show()


# # Function to plot actual vs predicted values for multiple symbols
# def plot_comparison(actual, predicted, symbols):
#     plt.figure(figsize=(15, 6))
#     for i in range(len(symbols)):
#         plt.plot(actual[i], label=f"Actual {symbols[i]}")
#         plt.plot(predicted[i], label=f"Predicted {symbols[i]}")
#     plt.ylabel("Adjusted Close (Scaled)")
#     plt.xlabel("Time Scale")
#     plt.legend()
#     plt.title("Actual vs Predicted Adjusted Close for Multiple Stocks")
#     plt.grid(True)
#     plt.show()


# # List of symbols to fetch and train on
# symbols = ["AAPL", "MSFT", "GOOGL"]
# actual_values = []
# predicted_values = []

# # Fetch data, train model, and make predictions for each symbol
# for symbol in symbols:
#     print(f"Processing data for {symbol}")
#     df = fetch_stock_data(symbol)

#     # Feature engineering
#     features = ["Open", "High", "Low", "Volume"]

#     # Scaling
#     scaler = StandardScaler()
#     X = scaler.fit_transform(df[features])
#     y = df["Adj Close"].values.reshape(-1, 1)

#     # Time Series Split
#     tscv = TimeSeriesSplit(n_splits=10)
#     for train_index, test_index in tscv.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]

#     # Reshape for LSTM input
#     X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
#     X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

#     # Train LSTM model
#     model = train_lstm_model(X_train, y_train, X_test, y_test)

#     # Make predictions
#     y_pred = model.predict(X_test)

#     actual_values.append(y_test)
#     predicted_values.append(y_pred)

# # Plot actual vs predicted values for each symbol
# for i in range(len(symbols)):
#     plot_predictions(
#         actual_values[i],
#         predicted_values[i],
#         f"Actual vs Predicted Adjusted Close for {symbols[i]}",
#     )

# # Plot comparison of actual vs predicted values for all symbols
# plot_comparison(actual_values, predicted_values, symbols)


from requests_html import HTMLSession
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from statsmodels.tsa.seasonal import seasonal_decompose


# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}/history?p={symbol}"
    session = HTMLSession()
    r = session.get(url)
    rows = r.html.xpath("//table/tbody/tr")
    data = []
    for row in rows:
        if len(row.xpath(".//td")) < 7:
            continue
        data.append(
            {
                "Symbol": symbol,
                "Date": row.xpath(".//td[1]/span/text()")[0],
                "Open": row.xpath(".//td[2]/span/text()")[0],
                "High": row.xpath(".//td[3]/span/text()")[0],
                "Low": row.xpath(".//td[4]/span/text()")[0],
                "Close": row.xpath(".//td[5]/span/text()")[0],
                "Adj Close": row.xpath(".//td[6]/span/text()")[0],
                "Volume": row.xpath(".//td[7]/span/text()")[0],
            }
        )
    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    str_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df[str_cols] = df[str_cols].replace(",", "", regex=True).astype(float)
    df.dropna(inplace=True)
    df = df.set_index("Date")

    # Additional feature engineering
    df["MACD_Histogram"] = macd_histogram(df)

    return df


def macd_histogram(df, short_window=12, long_window=26, signal_window=9):
    # Calculate short and long term EMAs
    short_ema = df["Close"].ewm(span=short_window, min_periods=1, adjust=False).mean()
    long_ema = df["Close"].ewm(span=long_window, min_periods=1, adjust=False).mean()

    # Calculate MACD line
    macd_line = short_ema - long_ema

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()

    # Calculate MACD histogram
    macd_histogram = macd_line - signal_line

    return macd_histogram


# Function to train LSTM model on stock data
def train_lstm_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(
        LSTM(
            32,
            activation="relu",
            return_sequences=True,
            input_shape=(1, X_train.shape[2]),
        )
    )
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation="relu", return_sequences=False))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.004))
    model.fit(
        X_train,
        y_train,
        epochs=300,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=0,
    )
    return model


# Function to plot actual vs predicted values
def plot_predictions(y_test, y_pred, title):
    plt.figure(figsize=(15, 6))
    plt.plot(y_test, label="Actual Value")
    plt.plot(y_pred, label="Predicted Value")
    plt.ylabel("Adjusted Close (Scaled)")
    plt.xlabel("Time Scale")
    plt.legend()
    plt.title(title)
    plt.show()


# List of symbols to fetch and train on
symbols = ["AAPL", "MSFT", "GOOGL"]

# Fetch data, train model, and make predictions for each symbol
for symbol in symbols:
    print(f"Processing data for {symbol}")
    df = fetch_stock_data(symbol)

    # Feature engineering
    features = ["Open", "High", "Low", "Volume", "MACD_Histogram"]

    # Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df["Adj Close"].values.reshape(-1, 1)

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    # Reshape for LSTM input
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Train LSTM model
    model = train_lstm_model(X_train, y_train, X_test, y_test)

    # Make predictions
    y_pred = model.predict(X_test)

    # Plot actual vs predicted values
    plot_predictions(y_test, y_pred, f"Actual vs Predicted Adjusted Close for {symbol}")

    # Plot MACD Histogram
    plt.figure(figsize=(15, 6))
    plt.bar(
        df.index,
        df["MACD_Histogram"],
        color=np.where(df["MACD_Histogram"] > 0, "g", "r"),
    )
    plt.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("Date")
    plt.ylabel("MACD Histogram")
    plt.title(f"MACD Histogram for {symbol}")
    plt.show()
