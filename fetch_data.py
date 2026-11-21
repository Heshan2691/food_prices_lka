from pathlib import Path
import urllib.request
import pandas as pd

DATA_URL = (
    "https://data.humdata.org/dataset/0298c598-d312-4771-b564-f4ac4d831f05/"
    "resource/3638f0d6-9969-48cf-a919-1d879d037ec6/download/wfp_food_prices_lka.csv"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "wfp_food_prices_lka.csv"


def main():
    print(f"Downloading official Sri Lanka Food Prices dataset from HDX...")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        OUT_FILE.write_bytes(resp.read())

    # HDX WFP files often include a 2nd row of HXL tags starting with '#'
    df = pd.read_csv(OUT_FILE, low_memory=False)
    if len(df) > 0 and str(df.iloc[0, 0]).startswith("#"):
        df = df.iloc[1:].reset_index(drop=True)
        df.to_csv(OUT_FILE, index=False)
        print("Removed HXL metadata header row.")

    print(f"Saved {len(df):,} rows x {df.shape[1]} columns to: {OUT_FILE}")
    print("Columns:", list(df.columns))
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
