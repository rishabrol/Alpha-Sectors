import requests
import pandas as pd
import time
from tqdm import tqdm

API_KEY = "8FZJLBAVPCF4TFMA"  # Replace with your actual key
CSV_PATH = "C:/Users/rishi/Desktop/project/Eligible companies.csv"
BASE_URL = "https://www.alphavantage.co/query"
OUTPUT_PATH = "C:/Users/rishi/Desktop/project/Eligible companies_updated_bd.csv"
def fetch_business_description(ticker):
    """Fetch the company overview and extract the description."""
    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": API_KEY
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("Description", "")
    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return ""

def get_all_descriptions(ticker_df):
    """Fetch descriptions for all tickers in a DataFrame."""
    results = []

    for i, row in tqdm(ticker_df.iterrows(), total=len(ticker_df), desc="Fetching Descriptions"):
        ticker = row['Ticker']
        company = row['Company']
        description = fetch_business_description(ticker)
        results.append({
            "Ticker": ticker,
            "Company": company,
            "Description": description
        })

        time.sleep(12.1)  # Respect Alpha Vantage rate limits

    return pd.DataFrame(results)
# === MAIN EXECUTION ===
if __name__ == "__main__":
    df_filtered = pd.read_csv(CSV_PATH)
    df_descriptions = get_all_descriptions(df_filtered)
    df_descriptions.to_csv(OUTPUT_PATH, index=False)
    print(f"Descriptions saved to {OUTPUT_PATH}")