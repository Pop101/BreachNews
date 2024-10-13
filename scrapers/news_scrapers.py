from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
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
        elif news_company.lower() == "cnn":
            return self._scrape_cnn(url)
        elif news_company.lower() == "fox":
            return self._scrape_fox(url)
        elif news_company.lower() == "washington post":
            return self._scrape_washington_post(url)
        elif news_company.lower() == "daily mail":
            return self._scrape_daily_mail(url)
        elif news_company.lower() == "cnbc":
            return self._scrape_cnbc(url)
        elif news_company.lower() == "the guardian":
            return self._scrape_the_guardian(url)
        elif news_company.lower() == "new york post":
            return self._scrape_new_york_post(url)
        elif news_company.lower() == "bbc":
            return self._scrape_bbc(url)
        elif news_company.lower() == "usa today":
            return self._scrape_usa_today(url)
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
    

    def _scrape_cnn(self, url):
        """
        Private method to scrape the CNN article given its URL.

        Args:
            url (str): The URL of the CNN article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "storyTD")))

            elements = full_html.find_elements(
                By.XPATH, ".//p | .//*[contains(@class, 'inStoryHeading')]"
            )

            full_html = full_html.get_attribute('outerHTML')

            article_html = ""
            for element in elements:
                article_html += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(article_html)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_fox(self, url):
        """
        Private method to scrape the Fox article given its URL.

        Args:
            url (str): The URL of the Fox article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "article-wrap")))

            elements = full_html.find_elements(
                By.XPATH, ".//p"
            )

            full_html = full_html.get_attribute('outerHTML')

            article_html = ""
            for element in elements:
                article_html += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(article_html)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_washington_post(self, url):
        """
        Private method to scrape the Washington Post article given its URL.

        Args:
            url (str): The URL of the Washington Post article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "grid-main-standard"))).get_attribute('outerHTML')

            article_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "grid-center")))

            elements = article_html.find_elements(
                By.XPATH, ".//p"
            )

            p_elements = ""
            for element in elements:
                p_elements += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(p_elements)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_daily_mail(self, url):
        """
        Private method to scrape the Daily Mail article given its URL.

        Args:
            url (str): The URL of the Daily Mail article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.XPATH, "//*[@itemprop='articleBody']")))

            elements = full_html.find_elements(
                By.XPATH, ".//p"
            )

            p_elements = ""
            for element in elements:
                p_elements += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(p_elements)
            full_html = full_html.get_attribute('outerHTML')

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_cnbc(self, url):
        """
        Private method to scrape the CNBC article given its URL.

        Args:
            url (str): The URL of the CNBC article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "PageBuilder-pageWrapper"))).get_attribute('outerHTML')

            article_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ArticleBody-articleBody")))

            elements = article_html.find_elements(
                By.XPATH, ".//p"
            )

            p_elements = ""
            for element in elements:
                p_elements += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(p_elements)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_the_guardian(self, url):
        """
        Private method to scrape the Guardian article given its URL.

        Args:
            url (str): The URL of the Guardian article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "dcr-1p5i1qs"))).get_attribute('outerHTML')

            article_html = wait.until(EC.presence_of_element_located((By.ID, "maincontent")))

            elements = article_html.find_elements(
                By.XPATH, ".//p"
            )

            p_elements = ""
            for element in elements:
                p_elements += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(p_elements)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_new_york_post(self, url):
        """
        Private method to scrape the New York Post article given its URL.

        Args:
            url (str): The URL of the New York Post article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "layout__inner")))

            elements = full_html.find_elements(
                By.XPATH, ".//p"
            )

            p_elements = ""
            for element in elements:
                p_elements += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(p_elements)
            full_html = full_html.get_attribute('outerHTML')

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_bbc(self, url):
        """
        Private method to scrape the BBC article given its URL.

        Args:
            url (str): The URL of the BBC article to scrape.

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

            # Wait for the article wrapper to load
            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ssrcss-15tkd6i-ArticleWrapper")))

            # Find the elements by their data-component attributes
            elements = None
            try:
                # Try locating elements using the XPATH selector for 'text-block' and 'subheadline-block'
                elements = driver.find_elements(By.XPATH, "//*[@data-component='text-block' or @data-component='subheadline-block']")

            except StaleElementReferenceException:
                print("Encountered a stale element. Re-fetching the elements...")

                # If stale element exception occurs, wait for the element again and retry finding elements
                full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "ssrcss-15tkd6i-ArticleWrapper")))
                elements = driver.find_elements(By.XPATH, "//*[@data-component='text-block' or @data-component='subheadline-block']")

            # Concatenate the text from the found elements
            article_html = ""
            for element in elements:
                article_html += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(article_html)
            full_html = full_html.get_attribute('outerHTML')

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit()  # Close the browser when done

        return article_text, full_html
    
    def _scrape_usa_today(self, url):
        """
        Private method to scrape the USA Today article given its URL.

        Args:
            url (str): The URL of the USA Today article to scrape.

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

            full_html = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "gnt_pr")))

            elements = full_html.find_elements(
                By.XPATH, ".//p"
            )

            p_elements = ""
            for element in elements:
                p_elements += element.get_attribute('outerHTML')

            article_text = self.remove_html_tags(p_elements)
            full_html = full_html.get_attribute('outerHTML')

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
# scraper = NewsScraper()
# url = "http://www.nytimes.com/2007/01/01/world/middleeast/01sunnis.html"
# article_text, full_html = scraper.scrape(url, "New York Times")
# print(article_text)

# scraper = NewsScraper()
# url = "http://money.cnn.com/2007/01/03/autos/new_chrysler_minivans/index.htm?cnn=yes"
# article_text, full_html = scraper.scrape(url, "CNN")
# print(article_text)

# scraper = NewsScraper()
# url = "https://www.foxnews.com/us/pregnant-woman-shot-in-the-head-while-driving-on-detroit-freeway"
# article_text, full_html = scraper.scrape(url, "Fox")
# print(article_text)

# scraper = NewsScraper()
# url = "https://www.washingtonpost.com/opinions/2024/10/13/trump-rally-interview-immigrants-lies/"
# article_text, full_html = scraper.scrape(url, "Washington Post")
# print(article_text)

# scraper = NewsScraper()
# url = "http://www.dailymail.co.uk/tvshowbiz/article-1308560/Katy-Perry-displays-unfortunate-underwear-choice-sheer-skirt.html"
# article_text, full_html = scraper.scrape(url, "Daily Mail")
# print(article_text)

# scraper = NewsScraper()
# url = "http://www.cnbc.com/id/16383511"
# article_text, full_html = scraper.scrape(url, "CNBC")
# print(article_text)

# scraper = NewsScraper()
# url = "http://www.theguardian.com/world/2013/jul/29/hillary-clinton-obama-lunch-2016"
# article_text, full_html = scraper.scrape(url, "The Guardian")
# print(article_text)

# scraper = NewsScraper()
# url = "https://nypost.com/2009/09/25/kirsten-dunce/"
# article_text, full_html = scraper.scrape(url, "New York Post")
# print(article_text)

# scraper = NewsScraper()
# url = "http://www.bbc.co.uk/news/business-12918761"
# article_text, full_html = scraper.scrape(url, "BBC")
# print(article_text)

# scraper = NewsScraper()
# url = "https://www.usatoday.com/story/entertainment/celebrities/2024/10/12/sean-diddy-combs-accuser-adria-english-responds-lawyers-withdraw/75648028007/"
# article_text, full_html = scraper.scrape(url, "USA Today")
# print(article_text)
