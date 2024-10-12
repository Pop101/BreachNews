from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

class NewsScraper:
    def __init__(self):
        # Set Chrome options to disable GPU and other settings
        self.chrome_options = Options()
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")

    def scrape(self, url, news_company):
        """
        Public method to scrape articles from different news companies based on the provided URL.

        Args:
            url (str): The URL of the news article to scrape.
            news_company (str): The name of the news company (e.g., 'New York Times').

        Returns:
            tuple: A tuple containing:
                - str: The cleaned text extracted from the article.
                - str: The full HTML content of the article.
        """
        if news_company.lower() == "new york times":
            return self._scrape_new_york_times(url)
        elif news_company.lower() == "another news company":
            return self._scrape_another_news_company(url)
        # Add more elif cases for other news companies here
        else:
            raise ValueError(f"No scraping method defined for {news_company}")

    def _scrape_new_york_times(self, url):
        """
        Private method to scrape the New York Times article given its URL.

        Args:
            url (str): The URL of the New York Times article to scrape.

        Returns:
            tuple: A tuple containing:
                - str: The cleaned text extracted from the article.
                - str: The full HTML content of the article.
        """
        driver = webdriver.Chrome(options=self.chrome_options)
        article_text = ""
        full_html = ""
        try:
            driver.get(url)
            wait = WebDriverWait(driver, 10)

            full_html = wait.until(EC.presence_of_element_located((By.ID, "story"))).get_attribute('outerHTML')

            # Use XPath to find all elements with data-testid that starts with "companionColumn-"
            elements = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//*[starts-with(@data-testid, 'companionColumn-')]")))

            article_html = ""
            for element in elements:
                paragraphs = element.find_elements(By.TAG_NAME, "p")
                for p in paragraphs:
                    article_html += p.get_attribute('outerHTML')

            article_text = self.remove_html_tags(article_html)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html

    @staticmethod
    def remove_html_tags(html):
        """
        Remove HTML tags from a given HTML string and return the cleaned text.
        """
        soup = BeautifulSoup(html, "html.parser")
        cleaned_text = ' '.join(soup.stripped_strings)
        return cleaned_text

# Example usage:
scraper = NewsScraper()
url = "http://www.nytimes.com/2007/01/01/world/middleeast/01sunnis.html"
article_text, full_html = scraper.scrape(url, "New York Times")
print(article_text)
