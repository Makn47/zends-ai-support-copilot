"""
00_scrape_imdb_optional.py

OPTIONAL data scraping script (not executed for this project — the brief
supplied `imdb_movies_2024.csv` directly, and that dataset was used for all
downstream steps). Included to satisfy the "Scraping Script" deliverable
and to document how the dataset could be regenerated/extended from IMDb
directly if a fresher or larger dataset is ever needed.

Scrapes movie name + storyline from IMDb's 2024 movies list using Selenium.

Requirements: `pip install selenium` and a matching Chromedriver on PATH.

Usage:
    python 00_scrape_imdb_optional.py --url "https://www.imdb.com/search/title/?title_type=feature&release_date=2024-01-01,2024-12-31" --output ../data/imdb_movies_2024_scraped.csv --max-movies 500
"""

import argparse
import csv
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options


def scrape_imdb_2024(url: str, max_movies: int = 500) -> list[dict]:
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)

    records = []
    try:
        driver.get(url)
        time.sleep(2)

        # Load more pages by scrolling / clicking "50 more" until max_movies reached.
        # IMDb's search results page lazy-loads; this loop clicks the "load more"
        # button (selector may need updating if IMDb's markup changes).
        while len(records) < max_movies:
            cards = driver.find_elements(By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
            for card in cards[len(records):]:
                try:
                    name = card.find_element(By.CSS_SELECTOR, "h3.ipc-title__text").text
                    storyline = card.find_element(
                        By.CSS_SELECTOR, ".ipc-html-content-inner-div"
                    ).text
                    records.append({"Movie Name": name, "Storyline": storyline})
                except Exception:
                    continue
                if len(records) >= max_movies:
                    break

            try:
                load_more = driver.find_element(By.CSS_SELECTOR, "button.ipc-see-more__button")
                load_more.click()
                time.sleep(2)
            except Exception:
                break  # no more pages
    finally:
        driver.quit()

    return records


def main():
    parser = argparse.ArgumentParser(description="Scrape IMDb 2024 movie names + storylines (optional)")
    parser.add_argument("--url", required=True, help="IMDb 2024 movies search URL")
    parser.add_argument("--output", default="../data/imdb_movies_2024_scraped.csv")
    parser.add_argument("--max-movies", type=int, default=500)
    args = parser.parse_args()

    records = scrape_imdb_2024(args.url, args.max_movies)
    print(f"Scraped {len(records)} movies")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Movie Name", "Storyline"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
