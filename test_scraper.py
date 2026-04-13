from amazon_scraper import scrape_amazon_reviews

url = "https://www.amazon.com/dp/B001EO5Q64"
reviews = scrape_amazon_reviews(url)
print("\n🔎 Final sample reviews:")
for review in reviews[:3]:
    print("-", review)