import time
import warnings

import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")


def fetch_page_content(url: str) -> BeautifulSoup:
    time.sleep(1)
    response = requests.get(url, verify=False)
    return BeautifulSoup(response.content, "html.parser")


def get_article_urls(BASE_URL: str, page_start: int) -> list:
    page_url = f"{BASE_URL}/knowledgebase/?pageStart={page_start}"
    soup = fetch_page_content(page_url)
    articles = soup.find_all("a", attrs={"class": "knowledgeBaseCard__title"})

    return [BASE_URL + article["href"] for article in articles]


def write_pdf_url(
    BASE_URL: str, WRITE_DIR: str, article_url: str, knowledge_base: list = []
):
    article_soup = fetch_page_content(article_url)
    doc_cards = article_soup.find_all("a", {"class": "docListCard"})
    if doc_cards:
        for card in doc_cards:
            if card["href"].endswith(".pdf"):
                pdf_url = BASE_URL + card["href"]
                if pdf_url + "\n" not in knowledge_base:
                    with open(WRITE_DIR, "a") as f:
                        f.write(pdf_url + "\n")


def main():
    BASE_URL = "https://ai.gov.ru"
    WRITE_DIR: str = "knowledge_base.txt"
    page_start = 0

    with open("knowledge_base.txt", "r") as f:
        knowledge_base = f.readlines()

    while True:
        article_urls = get_article_urls(BASE_URL, page_start)
        if article_urls:
            for article_url in article_urls:
                write_pdf_url(BASE_URL, WRITE_DIR, article_url, knowledge_base)
        else:
            break
        page_start += 10
        if page_start == 100:
            break


if __name__ == "__main__":
    main()
