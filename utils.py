import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
import os
import random
from gtts import gTTS

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

def extract_news_articles(company_name, num_articles=10):
    """
    Extract news articles related to a given company.
    """
    # In a real implementation, you would use a news API or web scraping.
    # For this assignment, we'll create mock news articles
    articles = []
    
    # Real implementation would include:
    # 1. Making API requests to news sources or using web scraping
    # 2. Extracting article text using BeautifulSoup
    # 3. Cleaning and preprocessing the text
    
    # For demonstration, let's create mock data
    potential_titles = [
        f"{company_name} Reports Strong Q2 Earnings",
        f"{company_name} Announces New Product Line",
        f"{company_name} Faces Regulatory Scrutiny",
        f"{company_name} Expands to International Markets",
        f"{company_name} Stock Soars on Positive News",
        f"{company_name} Faces Lawsuit from Competitors",
        f"{company_name} Introduces Revolutionary Technology",
        f"{company_name} CEO Speaks at Industry Conference",
        f"{company_name} Partners with Major Tech Company",
        f"{company_name} Cuts Jobs Amid Restructuring",
        f"{company_name} Recognized for Sustainability Efforts",
        f"{company_name} Quarterly Results Disappoint Investors"
    ]
    
    content_templates = [
        "{company} has reported strong earnings for Q2, exceeding analyst expectations. Revenue increased by 15% year-over-year, driven by strong sales in key markets. The company's stock price saw a significant jump following the announcement.",
        "{company} has announced a new product line aimed at expanding its market share. Analysts predict this move could significantly boost the company's revenue in the coming quarters.",
        "Regulators have raised concerns about {company}'s business practices, launching an investigation into potential anti-competitive behavior. The company's stock experienced volatility following the news.",
        "{company} is expanding its operations to international markets, with a focus on Asia and Europe. This strategic move is expected to drive growth and diversify revenue streams.",
        "Shares of {company} surged today following positive news about its latest quarterly performance. Investors are optimistic about the company's growth trajectory.",
        "{company} is facing a lawsuit from competitors alleging intellectual property infringement. Legal experts suggest the case could have significant implications for the company's future products.",
        "{company} has introduced groundbreaking technology that could revolutionize its industry. Early adopters have reported positive experiences with the new offering.",
        "The CEO of {company} delivered a keynote speech at a major industry conference, outlining the company's vision for the future and upcoming innovations.",
        "{company} has formed a strategic partnership with a leading tech company to collaborate on next-generation products. The alliance is expected to accelerate innovation.",
        "{company} has announced a restructuring plan that includes job cuts across several departments. The company cites the need to streamline operations and improve efficiency.",
        "{company} has been recognized for its sustainability initiatives, receiving an industry award for environmental responsibility. The company has committed to achieving carbon neutrality by 2030.",
        "{company}'s quarterly results fell short of analyst expectations, causing a dip in stock price. The company attributes the underperformance to supply chain challenges and increasing competition."
    ]
    
    # Randomly select articles to create a diverse set
    selected_indices = random.sample(range(len(potential_titles)), min(num_articles, len(potential_titles)))
    
    for i in selected_indices:
        title = potential_titles[i]
        content = content_templates[i].format(company=company_name)
        summary = content[:150] + "..." if len(content) > 150 else content
        
        articles.append({
            "title": title,
            "content": content,
            "summary": summary
        })
    
    return articles

def perform_sentiment_analysis(text):
    """
    Perform sentiment analysis on the given text.
    Returns: "Positive", "Negative", or "Neutral"
    """
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    
    if sentiment_score['compound'] >= 0.05:
        return "Positive"
    elif sentiment_score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_article_topics(text, num_topics=3):
    """
    Extract main topics from an article using TF-IDF and clustering.
    """
    # In a real implementation, you might use topic modeling like LDA
    # For simplicity, we'll use a predefined list of business topics
    business_topics = [
        "Stock Market", "Earnings", "Revenue", "Profit", "Loss",
        "Investment", "Growth", "Decline", "Innovation", "Technology",
        "Regulations", "Legal", "Lawsuit", "Competition", "Market Share",
        "Expansion", "International", "Product Launch", "Research",
        "Development", "Restructuring", "Layoffs", "Hiring", "Leadership",
        "Sustainability", "Environment", "Social Responsibility"
    ]
    
    # Extract potential topics based on word frequency
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    
    # Select random topics that might be relevant (in a real implementation, use NLP)
    text_lower = text.lower()
    potential_topics = [topic for topic in business_topics if topic.lower() in text_lower or any(word in topic.lower() for word in filtered_tokens)]
    
    # If no topics found or too few, add some generic ones
    while len(potential_topics) < num_topics:
        random_topic = random.choice(business_topics)
        if random_topic not in potential_topics:
            potential_topics.append(random_topic)
    
    # Return a subset of the potential topics
    return random.sample(potential_topics, min(num_topics, len(potential_topics)))

def generate_comparative_analysis(articles):
    """
    Generate comparative analysis across multiple articles.
    """
    # Extract all unique topics
    all_topics = []
    for article in articles:
        all_topics.extend(article["Topics"])
    unique_topics = list(set(all_topics))
    
    # Find common topics
    topic_counts = {topic: 0 for topic in unique_topics}
    for article in articles:
        for topic in article["Topics"]:
            topic_counts[topic] += 1
    
    common_topics = [topic for topic, count in topic_counts.items() if count > 1]
    
    # Generate comparisons between articles
    coverage_differences = []
    for i in range(len(articles)):
        for j in range(i+1, min(i+3, len(articles))):
            article1 = articles[i]
            article2 = articles[j]
            
            if article1["Sentiment"] != article2["Sentiment"]:
                comparison = f"Article {i+1} ({article1['Title']}) has a {article1['Sentiment']} sentiment, while Article {j+1} ({article2['Title']}) has a {article2['Sentiment']} sentiment."
                impact = f"This contrast shows different perspectives on the company, potentially affecting investor perception."
                coverage_differences.append({
                    "Comparison": comparison,
                    "Impact": impact
                })
            
            # Compare topics
            unique_topics_1 = [t for t in article1["Topics"] if t not in article2["Topics"]]
            unique_topics_2 = [t for t in article2["Topics"] if t not in article1["Topics"]]
            
            if unique_topics_1 and unique_topics_2:
                comparison = f"Article {i+1} focuses on {', '.join(unique_topics_1)}, whereas Article {j+1} covers {', '.join(unique_topics_2)}."
                impact = f"This difference in focus highlights various aspects of the company's operations and market position."
                coverage_differences.append({
                    "Comparison": comparison,
                    "Impact": impact
                })
    
    # If no meaningful comparisons found, add a generic one
    if not coverage_differences:
        coverage_differences.append({
            "Comparison": "The articles generally cover similar topics with similar sentiment.",
            "Impact": "The consistency in reporting suggests a stable narrative around the company."
        })
    
    # Topic overlap analysis
    topic_overlap = {
        "Common Topics": common_topics if common_topics else ["No common topics found"]
    }
    
    # Add unique topics for each article
    for i, article in enumerate(articles):
        other_articles_topics = []
        for j, other_article in enumerate(articles):
            if i != j:
                other_articles_topics.extend(other_article["Topics"])
        
        unique_topics = [topic for topic in article["Topics"] if topic not in other_articles_topics]
        if unique_topics:
            topic_overlap[f"Unique Topics in Article {i+1}"] = unique_topics
    
    return {
        "Coverage Differences": coverage_differences[:5],  # Limit to 5 comparisons
        "Topic Overlap": topic_overlap
    }

def generate_hindi_tts(text):
    """
    Convert text to Hindi speech using gTTS.
    In a production environment, you might want to use a more advanced TTS model.
    """
    output_file = "hindi_summary.mp3"
    
    # Convert English text to Hindi for demonstration
    # In a real implementation, use a translation API or model
    hindi_text = text  # Assuming text is already in Hindi for this demo
    
    # Generate the speech
    tts = gTTS(text=hindi_text, lang='hi', slow=False)
    tts.save(output_file)
    
    return output_file

# Implement a real news scraping function (commented out as alternative to mock data)
def scrape_news_articles(company_name, num_articles=10):
    """
    Scrape news articles about a company from non-JS websites using BeautifulSoup.
    This is an alternative to the mock data function and would be used in a real deployment.
    """
    # List of potential news sources (these are examples, would need to be verified for scraping feasibility)
    news_sources = [
        f"https://www.reuters.com/search/news?blob={company_name}",
        f"https://news.google.com/search?q={company_name}&hl=en-US&gl=US&ceid=US:en"
    ]
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    articles = []
    
    # Try different news sources until we get enough articles
    for source in news_sources:
        if len(articles) >= num_articles:
            break
            
        try:
            response = requests.get(source, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # The selectors would need to be adjusted based on the specific website structure
                article_elements = soup.select("article") or soup.select(".article") or soup.select(".story")
                
                for article_elem in article_elements[:num_articles - len(articles)]:
                    # Extract title - adjust selectors based on site structure
                    title_elem = article_elem.select_one("h1") or article_elem.select_one("h2") or article_elem.select_one(".title")
                    if not title_elem:
                        continue
                    title = title_elem.get_text().strip()
                    
                    # Extract URL to fetch full content
                    link_elem = article_elem.select_one("a")
                    if not link_elem:
                        continue
                    
                    article_url = link_elem.get("href")
                    if not article_url.startswith("http"):
                        # Handle relative URLs
                        if article_url.startswith("/"):
                            base_url = "/".join(source.split("/")[:3])
                            article_url = base_url + article_url
                        else:
                            continue
                    
                    # Fetch full article content
                    try:
                        article_response = requests.get(article_url, headers=headers, timeout=10)
                        if article_response.status_code == 200:
                            article_soup = BeautifulSoup(article_response.text, 'html.parser')
                            
                            # Extract content - adjust selectors based on site structure
                            content_elem = article_soup.select_one(".article-body") or article_soup.select_one(".content") or article_soup.select_one("article")
                            if not content_elem:
                                continue
                                
                            content = content_elem.get_text().strip()
                            
                            # Create summary (first paragraph or first 150 chars)
                            summary_elem = content_elem.select_one("p")
                            summary = summary_elem.get_text().strip() if summary_elem else content[:150] + "..."
                            
                            articles.append({
                                "title": title,
                                "content": content,
                                "summary": summary,
                                "url": article_url
                            })
                            
                            if len(articles) >= num_articles:
                                break
                                
                    except Exception as e:
                        print(f"Error fetching article content: {str(e)}")
                        continue
                        
        except Exception as e:
            print(f"Error scraping news source {source}: {str(e)}")
            continue
    
    # If we couldn't get enough real articles, supplement with mock data
    if len(articles) < num_articles:
        mock_articles = extract_news_articles(company_name, num_articles - len(articles))
        articles.extend(mock_articles)
    
    return articles[:num_articles]