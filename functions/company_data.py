import requests
from bs4 import BeautifulSoup

def fetch_company_website_text(url):
    """
    Fetches text from a website using requests + BeautifulSoup.
    Removes script/style content. Returns raw text.
    """
    try:
        resp = requests.get(url)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[ERROR] HTTP request failed for {url}: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text

def gather_portfolio_data(portfolio, portfolio_name="EQT_X_Portfolio"):
    """
    Visits each portfolio company's URL, scrapes basic text,
    and returns a dict {company_name: text_content}.
    """
    data = []
    for name, url in portfolio.items():
        print(f"[SCRAPE] Fetching data for {name} from {url}")
        text = fetch_company_website_text(url)
        data.append({
                "company_name": name,
                "url": url,
                "text": text, 
                "source": portfolio_name
            })
    return data

