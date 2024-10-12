from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def remove_html_tags(html):
    """
    Remove HTML tags from a given HTML string and return the cleaned text.

    This function uses BeautifulSoup to parse the HTML, extract text, 
    and remove extra whitespace by joining non-empty strings.

    Args:
        html (str): The HTML string from which to remove tags.

    Returns:
        str: The cleaned text without HTML tags, with extra spaces removed.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Get text and replace multiple spaces with a single space
    cleaned_text = ' '.join(soup.stripped_strings)
    return cleaned_text


def newYorkTimesArticle(url):
    """
    Retrieve the main text content and full HTML of a New York Times article 
    given its URL.

    This function initializes a Chrome WebDriver, navigates to the provided 
    URL, and waits for the necessary elements to load. It extracts the 
    HTML content from the specified elements, removes HTML tags from the 
    extracted paragraphs, and returns both the cleaned text and the full 
    HTML of the article.

    Args:
        url (str): The URL of the New York Times article to scrape.

    Returns:
        tuple: A tuple containing:
            - str: The cleaned text extracted from the article.
            - str: The full HTML content of the article element with ID "story".
    """
    # Set Chrome options to disable GPU
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize Chrome WebDriver with the options
    driver = webdriver.Chrome(options=chrome_options)

    article_text = ""
    full_html = ""
    try:
        # Navigate to the URL
        driver.get(url)

        # Wait for the page to load
        wait = WebDriverWait(driver, 10)

        full_html = wait.until(EC.presence_of_element_located((By.ID, "story"))).get_attribute('outerHTML')

        # Use XPath to find all elements with data-testid that starts with "companionColumn-"
        elements = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//*[starts-with(@data-testid, 'companionColumn-')]")))

        article_html = ""
        for element in elements:
            element_html = element.find_elements(By.TAG_NAME, "p")#element.get_attribute('outerHTML')
            for p in element_html:
                article_html += p.get_attribute('outerHTML')

        article_text = remove_html_tags(article_html)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()  # Close the browser when done

    return (article_text, full_html)



    