import requests
from bs4 import BeautifulSoup
from newspaper import Article

def fetch_news(company_name):
    """
    Fetches news articles related to a company from Bing News and extracts the article content.
    """
    search_url = f"https://www.bing.com/news/search?q={company_name}&FORM=HDRSC6"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return f"Failed to retrieve news. Status Code: {response.status_code}"

    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract news article links
    news_links = [a['href'] for a in soup.select("a[href]") if "http" in a['href']]
    
    articles = []
    
    for link in news_links[:10]:  # Limit to first 10 articles
        try:
            article = Article(link)
            article.download()
            article.parse()
            
            articles.append({
                "Title": article.title,
                "URL": link,
                "Summary": article.text[:500] + "..." if len(article.text) > 500 else article.text
            })
        except:
            continue

    return articles

# Example Usage:
if __name__ == "__main__":
    company = "Tesla"
    news_data = fetch_news(company)
    for article in news_data:
        print(article)
