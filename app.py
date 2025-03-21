streamlit run app.py
# ========== app.py ==========
import streamlit as st
import os
from fetch_news import fetch_news_with_sentiment  # Import function for news extraction
from sentiment_analysis import comparative_sentiment_analysis  # Import sentiment analysis function
from tts_hindi import text_to_speech  # Import Text-to-Speech function

# Streamlit UI
st.title("📰 News Summarization & Hindi TTS")
st.subheader("Analyze news sentiment and listen to Hindi summaries")

# User Input: Enter Company Name
company_name = st.text_input("Enter Company Name", "Tesla")

if st.button("Fetch News"):
    with st.spinner("Fetching news..."):
        news_articles = fetch_news_with_sentiment(company_name)  # Fetch news
    
    if not news_articles:
        st.error("No news articles found!")
    else:
        # Display Articles
        st.subheader(f"📰 News Articles for {company_name}")
        for article in news_articles:
            st.write(f"**{article['Title']}**")
            st.write(f"🔗 [Read More]({article['URL']})")
            st.write(f"📝 Summary: {article['Summary']}")
            st.write(f"📊 Sentiment: **{article['Sentiment']}**")
            st.write("---")

        # Perform Comparative Sentiment Analysis
        with st.spinner("Analyzing sentiment..."):
            analysis_result = comparative_sentiment_analysis(news_articles)

        st.subheader("📊 Comparative Sentiment Analysis")
        st.json(analysis_result)

        # Generate Hindi TTS Summary
        final_summary = f"{company_name} की खबरें ज्यादातर {max(analysis_result['Sentiment Distribution'], key=analysis_result['Sentiment Distribution'].get)} हैं।"
        audio_file = text_to_speech(final_summary, "summary.mp3")

        # Play Audio
        st.subheader("🔊 Hindi Audio Summary")
        st.audio(audio_file, format="audio/mp3")
