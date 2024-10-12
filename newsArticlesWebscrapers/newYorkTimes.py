from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def newYorkTimesArticle(url):
    # Set Chrome options to disable GPU
    chrome_options = Options()
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize Chrome WebDriver with the options
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Navigate to the URL
        driver.get(url)

        # Wait for the page to load
        wait = WebDriverWait(driver, 10)

        # Use XPath to find all elements with data-testid that starts with "companionColumn-"
        elements = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//*[starts-with(@data-testid, 'companionColumn-')]")))

        article_html = ""
        for element in elements:
            element_html = element.get_attribute('outerHTML')
            article_html += element_html
        print(article_html)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()  # Close the browser when done

# Example usage
newYorkTimesArticle("https://www.nytimes.com/2007/01/01/world/middleeast/01iraq.html?hp&ex=1167714000&en=85dae91ed8178e3a&ei=5094&partner=homepage")



    