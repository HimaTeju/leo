import streamlit as st
from logic import fetch_stock_data, preprocess_data, train_model, evaluate_model, plot_predictions, predict_next_day
import pandas as pd


def get_user_input():
    """Get user input for stock symbol and date range."""
    st.subheader("User Input")
    ticker = st.text_input("Enter Stock Symbol", "AAPL")
    start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-01"))
    return ticker, start_date, end_date


def main():
    """Main function for Streamlit app."""
    st.title("Stock Market Prediction")

    # Step 1: Get user input
    ticker, start_date, end_date = get_user_input()

    # Step 2: Fetch and display stock data
    df = fetch_stock_data(ticker, start_date, end_date)
    if df.empty:
        st.error("No data found for the given stock symbol and date range.")
        return

    st.subheader(f"Stock Data for {ticker}")
    st.write(df.tail())

    # Step 3: Preprocess data
    X, y = preprocess_data(df)

    # Step 4: Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Step 5: Train the model
    model = train_model(X_train, y_train)

    # Step 6: Evaluate the model
    y_pred, mse = evaluate_model(model, X_test, y_test)
    st.subheader("Model Evaluation")
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Step 7: Plot predictions
    st.subheader("Prediction vs Actual")
    plt = plot_predictions(df, y_test, y_pred)
    st.pyplot(plt)

    # Step 8: Predict next day's stock price
    next_day, predicted_price = predict_next_day(model, df.index[-1])
    st.subheader(f"Predicted Stock Price for {next_day}")
    st.write(f"Predicted price for {ticker}: ₹{predicted_price:.2f}")


if __name__ == "__main__":
    main()
