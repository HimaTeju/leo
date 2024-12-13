import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
import requests
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

# Streamlit App Configuration
st.set_page_config(page_title="Interactive Stock Dashboard", layout="wide")

# Sidebar: Stock Input
st.sidebar.title("Stock Selection")
stock_name = st.sidebar.text_input("Enter Stock Symbol", value="reliance.ns", max_chars=20).upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Validate Dates
if start_date >= end_date:
    st.error("Start date must be earlier than the end date. Please adjust the date range.")
    st.stop()

# Fetch Stock Data
@st.cache_data
def fetch_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        return str(e)

stock_data = fetch_stock_data(stock_name, start_date, end_date)

if stock_data is None or stock_data.empty:
    st.error(f"No data available for {stock_name}. Please check the stock symbol or date range.")
    st.stop()

# Add Sentiment column with random scores for demo
stock_data['Sentiment'] = np.random.uniform(-1, 1, len(stock_data))
buy_signals = stock_data.iloc[::15].index
sell_signals = stock_data.iloc[::20].index

# Tabs
tab_home, tab_signals, tab_trends, tab_analysis = st.tabs([
    "Home", "Buy/Sell Signals", "Predicted Trends", "Sentiment Analysis"
])

# Home Tab
with tab_home:
    st.title("📈 Stock Dashboard ")
    st.write(f"Displaying stock data for **{stock_name}** from {start_date} to {end_date}.")
    st.dataframe(stock_data)

# Buy/Sell Signals Tab
with tab_signals:
    st.title("💹 Buy/Sell Signals")
    buy_df = pd.DataFrame({"Date": list(buy_signals), "Signal": ["Buy"] * len(buy_signals)})
    sell_df = pd.DataFrame({"Date": list(sell_signals), "Signal": ["Sell"] * len(sell_signals)})
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Buy Signals")
        st.dataframe(buy_df)
    with col2:
        st.markdown("### Sell Signals")
        st.dataframe(sell_df)

# Predicted Trends Tab
with tab_trends:
    st.title("📊 Predicted Trends")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name="Stock Price"))
    fig_trend.update_layout(title=f"Predicted Stock Trend for {stock_name}", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig_trend, use_container_width=True)

# Sentiment Analysis Tab
with tab_analysis:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    def fetch_news(stock):
        company_name = stock.split('.')[0]
        url = f"https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey=310034844e9c446fa18c29a0e0b19b59"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    'title': article['title'],
                    'description': article['description'],
                    'publishedAt': article['publishedAt']
                } for article in data['articles']
            ]
        else:
            return []

    def preprocess_text(text):
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return " ".join(filtered_words)

    def analyze_sentiment(text):
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return scores['compound']

    st.title("📜 Sentiment Analysis of News Articles")
    st.write(f"Analyzing public sentiment for **{stock_name}** based on recent news articles.")
    news_articles = fetch_news(stock_name)

    if news_articles:
        analyzed_data = []
        for article in news_articles:
            title_desc = article['title'] + " " + article['description']
            preprocessed_text = preprocess_text(title_desc)
            sentiment_score = analyze_sentiment(preprocessed_text)
            sentiment_label = (
                "Positive" if sentiment_score > 0.2 else
                "Neutral" if -0.2 <= sentiment_score <= 0.2 else
                "Negative"
            )
            analyzed_data.append({"Article": title_desc, "Score": sentiment_score, "Sentiment": sentiment_label})

        df = pd.DataFrame(analyzed_data)
        st.dataframe(df)

        # Word Cloud
        st.write("### Word Cloud from News Articles:")
        wordcloud = WordCloud(width=800, height=400, background_color="black").generate(" ".join(df["Article"]))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)

        # Pie Chart for Sentiment Distribution
        st.write("### Sentiment Distribution:")
        sentiment_counts = df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]
        pie_chart = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            title="Sentiment Distribution",
            color="Sentiment",
            color_discrete_map={"Positive": "#2E8B57", "Neutral": "#FFD700", "Negative": "#FF6347"},
            hole=0,
        )
        st.plotly_chart(pie_chart)

        # Overall Statistics
        st.write("### Overall Sentiment Statistics:")
        sentiment_backgrounds = {
            "Positive": "#d4edda",  # Greenish background
            "Neutral": "#ffeeba",   # Yellowish background
            "Negative": "#f8d7da"   # Reddish background
        }
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"<div style='background-color:green;padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
            st.metric("Total Articles", len(news_articles))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='background-color:{sentiment_backgrounds['Positive']};padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
            st.metric("Positive Sentiment", len(df[df['Sentiment'] == 'Positive']))
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div style='background-color:{sentiment_backgrounds['Neutral']}; padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
            st.metric("Neutral Sentiment", len(df[df['Sentiment'] == 'Neutral']))
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(f"<div style='background-color:{sentiment_backgrounds['Negative']};padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
            st.metric("Negative Sentiment", len(df[df['Sentiment'] == 'Negative']))
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No news articles found.")
