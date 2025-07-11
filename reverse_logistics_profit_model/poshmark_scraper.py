# poshmark_scraper.py

"""
PoshmarkScraper Package
-----------------------

A Python module to randomly sample clothing listings from Poshmark and scrape key attributes.

Usage:
    from poshmark_scraper import PoshmarkScraper

    scraper = PoshmarkScraper(headless=True)
    scrape_dict = {           //dict = {Department: [num unsold, num sold]}
        "Women":[2565, 1285],
        "Men":[500, 250],
        "Kids":[265, 135]
    }
    scraper.collect_listings(dict=scrape_dict, max_pages=50000)
    df = scraper.scrape()
    print(df)
    df.to_csv("poshmark_sample.csv", index=False)
"""

import random
import time
import csv
from typing import List, Dict
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


class PoshmarkScraper:
    def __init__(self, headless: bool = True, wait: float = 2.5):
        """Initialize the Selenium WebDriver."""
        chrome_opts = Options()
        if headless:
            chrome_opts.add_argument("--headless")
        chrome_opts.add_argument("--disable-gpu")
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_opts)
        self.wait = wait

    def get_listing_links(self, page_url: str) -> List[str]:
        """Return all product URLs from a category page."""
        self.driver.get(page_url)
        time.sleep(self.wait)
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        anchors = soup.select('a[class="tile__title tc--b"]')
        print(len(anchors))
        full_links: List[str] = []
        for a in anchors:
            href = a.get("href")
            if href:
                # ensure href is treated as text
                href_str = str(href)
                # safer than simple ‘+’, and handles any leading slashes
                full_links.append(urljoin("https://poshmark.com", href_str))
        return full_links   

    def sample_listings(
        self, n_samples: int, max_pages: int, category_url: str
    ) -> List[str]:
        """
        Randomly sample listing URLs by visiting random pages under category_url.
        """
        sampled = set()
        pages = list(range(1, max_pages + 1))
        while len(sampled) < n_samples:
            page = random.choice(pages)
            url = f"{category_url}&page={page}"
            print(url)
            try:
                links = self.get_listing_links(url)
                print("trying to get link")
            except Exception:
                continue
            if links:
                sampled.update(random.choices(links, k=24))
                print(len(sampled), " link(s) added")
        return list(sampled)

    def parse_listing(self, url: str) -> Dict:
        """Scrape required fields from a single listing page."""
        self.driver.get(url)
        time.sleep(self.wait)
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        data: Dict = {"url": url}

        # Listing Price
        lp = soup.select_one('p[class="h1"]')
        data["listing_price"] = None

        if lp is not None:
            raw_txt = lp.get_text(strip=True)
            str_list = str(raw_txt).split("$")
            try: 
                data["listing_price"] = str_list[1]
            except IndexError:
                print("Indexing Error: Price not found")
            else:
                if len(str_list) > 2:
                    data["price_drop"] = str_list[2]

        # Price Drop
        if "price_drop" not in data or not data["price_drop"]:
            data["price_drop"] = None
            drop = soup.find(string=lambda txt: isinstance(txt, str) and "price dropped" in txt.lower())

            if drop is not None:
                raw_txt = drop.get_text(strip=True)
                data["price_drop"] = str(raw_txt).split()[2]
            

        # Discounted Shipping
        data["discounted_shipping"] = None
        ship = soup.select_one('span[style="color: #822432;"]')
        
        if ship is None: # Check if there's no discount
            ship = soup.find(string=lambda txt: isinstance(txt, str) and "$8.27" in txt.lower())
        
        if ship is not None:
            raw_txt = ship.get_text(strip=True)
            data["discounted_shipping"] = str(raw_txt).split()[0]

        # Style Tags
        tags = [t.get_text(strip=True) for t in soup.select('div[data-et-prop-style_tag]')]
        data["style_tags"] = tags

        # Color
        col = [t.get_text(strip=True) for t in soup.select('a[class="btn--icon tc--lg"]')]
        data["color"] = col

        # Brand
        br = soup.select_one('a[class="listing__brand listing__ipad-centered d--fl m--t--2"]')
        data["brand"] = br.get_text(strip=True) if br else None

        # New w/ Tags
        nw = soup.select_one('span[class="condition-tag all-caps tr--uppercase"]')
        data["new_with_tags"] = bool(nw and "nwt" in nw.get_text(strip=True).lower())

        # Size
        sz = [t.get_text(strip=True) for t in soup.select('div[class="d--fl fw--w ai--fs m--t--3 listing__size-selector-con"] button')]
        data["size"] = sz
           
        # Category & Sub-category
        data["sub_category"] = None
        data["category"] = None
        parent = soup.select_one('div[class="m--t--3 m--r--7"]') # Condition: Category is before Color

        if parent is not None:
            children = parent.find_all("div", recursive=False)
            cat = [t.get_text(strip=True) for t in children]
            try: 
                data["category"] = cat[:2]
            except IndexError:
                print("Category not found")
            else:
                if len(children) > 2:
                    data["sub_category"] = cat[2]

        # Discount Offer
        disc = soup.find(string=lambda txt: isinstance(txt, str) and "seller discount:" in txt.lower())
        data["bundle"] = bool(disc)

        # Listing Date via image URL hack
        img = soup.select_one('img[imageclass="img__container img__container--square"]')
        data["listing_date"] = None

        if img is not None:
            raw_src = img.get("src")
            src_str = str(raw_src) if raw_src is not None else ""

            if "cloudfront.net/posts/" in src_str:
                try: 
                    parts = src_str.split("/posts/")[1].split("/")[:3]
                except IndexError:
                    print("Date not found")
                else:
                    data["listing_date"] = "-".join(parts)
        

        # Sold status
        sold = soup.select_one('h1[class="listing__status-banner__title--sold"]')
        data["sold"] = bool(sold and "sold" in sold.get_text(strip=True).lower())

        return data

    def collect_listings(
        self, dict: Dict, max_pages: int
    ):
        """
        Sample n_samples URLs and scrape each one.
        Returns a pandas DataFrame of the results.
        """
        base_url = "https://poshmark.com/category/"
        sold = "?availability=sold_out"
        available = "?availability=available"
        links = []

        for key in dict:
            url = base_url + key + available
            links.extend(self.sample_listings(dict[key][0], max_pages, url))
            print("Available listings for " + key + " collected")
            sold_url = base_url + key + sold
            links.extend(self.sample_listings(dict[key][1], max_pages, sold_url))
            print("Sold listings for " + key + " collected")
        
        links_df = pd.DataFrame(links)
        links_df.to_csv("./data/raw/poshmark_urls.csv", index=False)

    def scrape(self) -> pd.DataFrame:
        rows = []
        filename = "./data/raw/poshmark_urls.csv"
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                try:
                    rows.append(self.parse_listing(row[0]))
                    print(len(rows), " row(s) appended")
                except Exception as e:
                    print(f"Error on {row}: {e}")
        self.driver.quit()
        return pd.DataFrame(rows)


if __name__ == "__main__":
    scraper = PoshmarkScraper(headless=True)
    scrape_dict = {
        "Women":[2565, 1285],
        "Men":[500, 250],
        "Kids":[265, 135]
    }
    scraper.collect_listings(dict=scrape_dict, max_pages=50000)
    df = scraper.scrape()
    print(df)
    df.to_csv("./data/raw/poshmark_sample.csv", index=False)

