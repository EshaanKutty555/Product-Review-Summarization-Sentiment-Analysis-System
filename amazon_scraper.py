from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

def scrape_amazon_reviews(product_url, max_reviews=20):
    print("🔗 Opening product page:", product_url)

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service(executable_path="chromedriver.exe")
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(product_url)
        print("✅ Page loaded")
        time.sleep(2)

        # Scroll to bottom to trigger lazy load
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Try clicking "See all reviews" link if available
        try:
            see_all = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'reviews')]"))
            )
            print("🔘 'See all reviews' link found:", see_all.text)
            see_all.click()
            print("🔁 Clicked to load full reviews page")
            time.sleep(2)
        except Exception as e:
            print("⚠️ Could not click reviews link, scraping top reviews:", e)

        # Wait for review blocks to appear
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-hook='review']"))
            )
            print("✅ Reviews detected on page")
        except Exception as e:
            print("⚠️ No reviews found after waiting:", e)

        # Parse page HTML
        soup = BeautifulSoup(driver.page_source, "html.parser")
        reviews = []

        for block in soup.select("[data-hook='review']"):
            review_body = block.select_one("[data-hook='review-body']")
            if review_body:
                text = review_body.get_text(strip=True)
                reviews.append(text)
                print("📝 Review:", text[:90])
            if len(reviews) >= max_reviews:
                break

        print(f"✅ Collected {len(reviews)} reviews")
        return reviews

    except Exception as e:
        print("❌ Unexpected error:", str(e))
        return []

    finally:
        driver.quit()
        print("🚪 Browser closed")
