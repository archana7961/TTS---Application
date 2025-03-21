from textblob import TextBlob
import nltk
from collections import Counter

# Download necessary components for TextBlob
nltk.download('punkt')

def get_sentiment(text):
    """
    Determines the sentiment of a given text using TextBlob.
    Returns 'Positive', 'Negative', or 'Neutral' based on polarity score.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Polarity ranges from -1 to 1

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def comparative_sentiment_analysis(news_articles):
    """
    Performs comparative sentiment analysis on multiple news articles.
    Returns a sentiment distribution and highlights differences in coverage.
    """
    sentiments = [article["Sentiment"] for article in news_articles]
    sentiment_counts = Counter(sentiments)

    analysis = {
        "Sentiment Distribution": {
            "Positive": sentiment_counts.get("Positive", 0),
            "Negative": sentiment_counts.get("Negative", 0),
            "Neutral": sentiment_counts.get("Neutral", 0)
        },
        "Coverage Differences": []
    }

    # Compare articles to identify differences in sentiment
    for i in range(len(news_articles) - 1):
        for j in range(i + 1, len(news_articles)):
            comparison = {
                "Article 1": news_articles[i]["Title"],
                "Article 2": news_articles[j]["Title"],
                "Comparison": f"Article 1 has a {news_articles[i]['Sentiment']} tone, while Article 2 has a {news_articles[j]['Sentiment']} tone."
            }
            analysis["Coverage Differences"].append(comparison)

    return analysis

# Example Usage:
if __name__ == "__main__":
    # Sample data
    sample_articles = [
        {"Title": "Tesla's Stock Soars", "Summary": "Tesla's stock price increased by 15% today.", "Sentiment": get_sentiment("Tesla's stock price increased by 15% today.")},
        {"Title": "Tesla Faces Legal Issues", "Summary": "Tesla is under investigation for regulatory issues.", "Sentiment": get_sentiment("Tesla is under investigation for regulatory issues.")}
    ]

    print("Sentiment Analysis Results:")
    print(sample_articles)

    print("\nComparative Analysis:")
    print(comparative_sentiment_analysis(sample_articles))
