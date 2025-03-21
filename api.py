from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import base64
import os
from utils import (
    extract_news_articles,
    perform_sentiment_analysis,
    generate_comparative_analysis,
    get_article_topics,
    generate_hindi_tts
)

app = FastAPI(title="News Sentiment Analysis API")

class CompanyRequest(BaseModel):
    company_name: str

@app.post("/analyze")
async def analyze_company(request: CompanyRequest):
    """
    Analyze news articles for a specified company.
    Returns sentiment analysis, comparative analysis, and Hindi TTS audio.
    """
    try:
        company_name = request.company_name
        
        # Extract news articles
        articles = extract_news_articles(company_name)
        
        # Perform sentiment analysis and topic extraction
        processed_articles = []
        sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
        all_topics = []
        
        for article in articles:
            # Get sentiment
            sentiment = perform_sentiment_analysis(article["content"])
            sentiments[sentiment] += 1
            
            # Get topics
            topics = get_article_topics(article["content"])
            all_topics.extend(topics)
            
            processed_article = {
                "Title": article["title"],
                "Summary": article["summary"],
                "Sentiment": sentiment,
                "Topics": topics
            }
            processed_articles.append(processed_article)
        
        # Generate comparative analysis
        comparative_analysis = generate_comparative_analysis(processed_articles)
        
        # Determine final sentiment
        if sentiments["Positive"] > sentiments["Negative"]:
            final_sentiment = f"{company_name}'s latest news coverage is mostly positive. Potential stock growth expected."
        elif sentiments["Positive"] < sentiments["Negative"]:
            final_sentiment = f"{company_name}'s latest news coverage is mostly negative. Caution advised."
        else:
            final_sentiment = f"{company_name}'s latest news coverage is mixed. Monitor developments closely."
        
        # Generate Hindi TTS
        hindi_summary = f"{company_name} के बारे में समाचार विश्लेषण। {final_sentiment}"
        audio_file = generate_hindi_tts(hindi_summary)
        
        # Read the audio file and encode as base64
        with open(audio_file, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode()
        
        # Prepare the response
        response = {
            "Company": company_name,
            "Articles": processed_articles,
            "Comparative Sentiment Score": {
                "Sentiment Distribution": sentiments,
                "Coverage Differences": comparative_analysis["Coverage Differences"],
                "Topic Overlap": comparative_analysis["Topic Overlap"]
            },
            "Final Sentiment Analysis": final_sentiment,
            "Audio": audio_base64
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "service": "News Sentiment Analysis API"}

@app.get("/companies")
async def get_sample_companies():
    """
    Returns a list of sample companies for demo purposes.
    """
    sample_companies = [
        "Apple", "Google", "Microsoft", "Amazon", "Tesla", 
        "Facebook", "Netflix", "IBM", "Intel", "Samsung"
    ]
    return {"companies": sample_companies}

@app.get("/")
async def root():
    """
    Root endpoint with API information and documentation.
    """
    return {
        "name": "News Sentiment Analysis API",
        "version": "1.0.0",
        "description": "API for analyzing news articles, sentiment, and generating Hindi TTS",
        "endpoints": {
            "/analyze": "POST - Analyze news for a company",
            "/health": "GET - Health check",
            "/companies": "GET - List of sample companies",
            "/docs": "OpenAPI documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)