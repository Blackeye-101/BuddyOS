import logging
import yfinance as yf
import requests
from core.tools import plugin

logger = logging.getLogger(__name__)

# Create a customized session to bypass aggressive Yahoo Finance blocking
# By spoofing a real browser's user-agent and accept headers, yfinance can reliably pull deep data.
_session = requests.Session()
_session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
})

@plugin(
    name="get_live_stock_info",
    description="Fetch deep stock fundamentals, daily pricing, and the latest news for Indian NSE/BSE tickers. Returns INR pricing. Use .NS for NSE or .BO for BSE.",
    allowed_agents=["finance_matcher", "finance_scraper"],
    parameters={
        "ticker": {"type": "string", "description": "The stock ticker symbol (e.g., 'RELIANCE.NS')"}
    }
)
def get_live_stock_info(ticker: str) -> str:
    """Fetch live stock details using yfinance with an anti-blocking session."""
    logger.info(f"Fetching deep live stock info for ticker: {ticker}")
    
    ticker_upper = ticker.upper()
    if not ticker_upper.endswith(".NS") and not ticker_upper.endswith(".BO"):
        ticker_upper = f"{ticker_upper}.NS"
        
    try:
        stock = yf.Ticker(ticker_upper, session=_session)
        info = stock.info
        
        if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
            # Attempt to fallback to fast info
            fast_info = stock.fast_info
            if hasattr(fast_info, 'last_price') and fast_info.last_price is not None:
                 current_price = round(fast_info.last_price, 2)
                 previous_close = round(fast_info.previous_close, 2)
                 change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)
                 return (
                     f"Ticker: {ticker_upper}\n"
                     f"Current Price: {current_price}\n"
                     f"Daily Change: {change_pct}%\n"
                     f"Note: Deep fundamentals blocked by provider."
                 )
            return f"Error: Could not retrieve data for {ticker_upper}. It might be delisted or invalid."
            
        current_price = info.get("currentPrice", info.get("regularMarketPrice", "N/A"))
        previous_close = info.get("previousClose", 0)
        
        change_pct = "N/A"
        if current_price != "N/A" and previous_close != 0:
            change_pct = round(((current_price - previous_close) / previous_close) * 100, 2)
            
        # Compile latest news
        recent_news = []
        try:
            news_items = stock.news[:3] # Get top 3 news items
            for n in news_items:
                headline = n.get('content', {}).get('title', '')
                if headline:
                    recent_news.append(f"- {headline}")
        except Exception:
            pass
        news_str = "\n".join(recent_news) if recent_news else "No recent news available."
            
        summary = (
            f"Ticker: {ticker_upper}\n"
            f"Company: {info.get('shortName', info.get('longName', 'Unknown'))}\n"
            f"Current Price: {info.get('currency', 'INR')} {current_price}\n"
            f"Previous Close: {previous_close} | Daily Change: {change_pct}%\n"
            f"52 Week Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"Market Cap: {info.get('marketCap', 'N/A')}\n"
            f"PE Ratio (Trailing): {info.get('trailingPE', 'N/A')} | Forward PE: {info.get('forwardPE', 'N/A')}\n"
            f"Dividend Yield: {info.get('dividendYield', 'N/A')}\n"
            f"Debt to Equity: {info.get('debtToEquity', 'N/A')}\n"
            f"Return on Equity (ROE): {info.get('returnOnEquity', 'N/A')}\n"
            f"Analyst Recommendation: {info.get('recommendationKey', 'N/A').upper()}\n"
            f"Target Price (1y): {info.get('targetMeanPrice', 'N/A')}\n"
            f"Business Summary: {info.get('longBusinessSummary', 'N/A')[:400]}...\n\n"
            f"Recent News:\n{news_str}"
        )
        return summary
    except Exception as e:
        logger.error(f"Failed to fetch stock info for {ticker}: {e}")
        return f"Error connecting to stock API for {ticker_upper}: {str(e)}"
