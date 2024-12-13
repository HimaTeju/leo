import streamlit as st
import requests
import re
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap

# Download NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Fetch News Data
def fetch_news(stock):
    # Extract the company name from stock (e.g., "company.NS" or "company.BS")
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
        st.error(f"Error fetching news: {response.status_code}")
        return []

# Preprocess Text
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Analyze Sentiment
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']

# Map Sentiment Score to Scale
def map_sentiment_to_scale(score):
    if score <= -0.2:
        return "Strong Sell", "Negative", int((1 + score) * 20)
    elif -0.2 < score < 0.2:
        return "Hold", "Neutral", int((score + 0.5) * 40)
    else:
        return "Strong Buy", "Positive", int((score * 40) + 60)

# Streamlit App
st.title("Stock Sentiment Analysis Tool")
st.markdown("Analyze public sentiment for a stock using recent news articles, with engaging visuals.")

# Input section
stock = st.text_input("Enter the stock name (e.g., company.NS or company.BS):")
if st.button("Analyze Sentiment"):
    if not stock.strip():
        st.error("Please enter a valid stock name.")
    elif not re.match(r'^[a-zA-Z]+\.(NS|BS)$', stock.strip()):
        st.error("Invalid format. Please use the format 'company.NS' or 'company.BS'.")
    else:
        # Fetch news
        st.write(f"Fetching news for **{stock}**...")
        news_articles = fetch_news(stock)

        if news_articles:
            # Preprocess and analyze
            st.write("Analyzing sentiment of news articles...")
            analyzed_data = []
            for article in news_articles:
                title_desc = article['title'] + " " + article['description']
                preprocessed_text = preprocess_text(title_desc)
                sentiment_score = analyze_sentiment(preprocessed_text)
                sentiment_label, sentiment_category, sentiment_value = map_sentiment_to_scale(sentiment_score)
                analyzed_data.append({
                    "Article": title_desc,
                    "Published At": article['publishedAt'],
                    "Sentiment": sentiment_label,
                    "Category": sentiment_category,
                    "Score": sentiment_value
                })

            df = pd.DataFrame(analyzed_data)
            df['Published At'] = pd.to_datetime(df['Published At']).dt.date

            # Filter for current day
            current_day = datetime.now().date()
            current_day_data = df[df['Published At'] == current_day]

            # Display all data
            st.write("### Sentiment Analysis Results:")
            st.dataframe(df)

            # Display current day sentiment
            st.write("### Sentiment for Today:")
            if not current_day_data.empty:
                st.dataframe(current_day_data)
                overall_score = current_day_data['Score'].mean()
                overall_label, overall_category, _ = map_sentiment_to_scale(overall_score)
                st.metric("Overall Sentiment", overall_label, delta=f"Category: {overall_category}")
                st.markdown("<style>.stMetric div {background-color: #e0e0e0; padding: 5px; border-radius: 5px;}</style>", unsafe_allow_html=True)
            else:
                st.warning("No articles found for today.")

            # Word Cloud
            # Custom colormap with green, blue, and red shades
            colors = ["green", "yellow", "red"]  # Green, Blue, Red
            custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

        # Word Cloud
            st.write("### Word Cloud from News Articles:")
            all_text = " ".join(df['Article'])
            wordcloud = WordCloud(width=800, height=400, background_color='black', colormap=custom_cmap).generate(all_text)

            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            # Pie Chart for Sentiment Distribution
            st.write("### Sentiment Distribution:")
            sentiment_counts = df["Category"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            pie_chart = px.pie(sentiment_counts, values="Count", names="Sentiment",
                               title="Sentiment Distribution",
                               color="Sentiment",
                               color_discrete_map={"Positive": "#2E8B57", "Neutral": "#FFD700", "Negative": "#FF6347"},
                               hole=0.4,
                               template="plotly_dark",
                               width=600,  # Adjust width
                               height=400)  # Adjust height
            st.plotly_chart(pie_chart)

            # Bar Chart for Sentiment Scores
            st.write("### Sentiment Scores by Article:")
            bar_chart = px.bar(df, x="Article", y="Score", color="Category",
                               title="Sentiment Scores by Article",
                               color_discrete_map={"Positive": "#2E8B57", "Neutral": "#FFD700", "Negative": "#FF6347"},
                               text_auto=True,
                               template="plotly_dark",
                               width=1000,  # Adjust width
                               height=600)  # Adjust height
            st.plotly_chart(bar_chart)

            # Summary Statistics
            st.write("### Summary Statistics:")
            sentiment_backgrounds = {
                "Positive": "#d4edda",  # Greenish background for positive
                "Neutral": "#ffeeba",   # Yellow background for neutral
                "Negative": "#f8d7da"   # Reddish background for negative
            }

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"<div style='background-color:green;padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.metric("Total Articles", len(news_articles))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"<div style='background-color:{sentiment_backgrounds['Positive']};padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.metric("Positive Sentiment", len(df[df['Category'] == 'Positive']))
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown(f"<div style='background-color:{sentiment_backgrounds['Neutral']}; padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.metric("Neutral Sentiment", len(df[df['Category'] == 'Neutral']))
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"<div style='background-color:{sentiment_backgrounds['Negative']};padding:10px; border-radius: 10px;'>", unsafe_allow_html=True)
                st.metric("Negative Sentiment", len(df[df['Category'] == 'Negative']))
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("No news articles found!")
