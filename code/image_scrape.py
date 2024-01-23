from selenium import webdriver
from selenium.webdriver.common.by import By

import os
import re
import time
import urllib

if __name__ == "__main__":
    image_urls = ...
    website_urls = ...
    driver = webdriver.Chrome()

    for website in website_urls:
        driver.maximize_window()
        driver.get(website)
        time.sleep(2)

        next_page = 1
        last_page = ...
        next_page_exists = True

        while next_page_exists and next_page != last_page:
            current_height = 0
            max_height = driver.execute_script("return document.body.scrollHeight;")

            while (max_height - current_height) != 0:
                for height in list(range(current_height, max_height, 10)) + [max_height]:
                    driver.execute_script(f"window.scrollTo(0,{height});")
                    time.sleep(1e-2)

                current_height = max_height
                max_height = driver.execute_script("return document.body.scrollHeight;")

            image_tags = driver.find_elements(By.TAG_NAME, "img")
            for tag in image_tags:
                if "ssense" in website or "saksoff5th" in website:
                    image_urls.append(tag.get_attribute("srcset")
                else:
                    image_urls.append(tag.get_attribute("src"))

            next_page = next_page + 1
            next_page_exists = False

            anchor_tags = driver.find_elements(By.TAG_NAME, 'a')
            for anchor_tag in anchor_tags:
                href = anchor_tag.get_attribute("href")
                if isinstance(href, str):
                    if ("page=" + str(next_page) in href) or ("pageNumber=" + str(next_page) in href):
                        next_page_exists = True
                        driver.execute_script("arguments[0].click();", anchor_tag)
                        time.sleep(2)
                        break

    image_urls = [url for url in image_urls if url is not Ellipsis and url is not None]
    image_urls = [url for url in image_urls if ".png" in url or ".jpg" in url or ".jpeg" in url]
    dataset = ...
    category = ...

    train_fol = os.getcwd() + "/../data/train" + category
    valid_fol = os.getcwd() + "/../data/valid" + category
    existing_files = os.listdir(train_fol) + os.listdir(valid_fol)

    for url in image_urls:
        file_to_request = re.search("[^/]+\.+(jpeg|jpg|png)+", url).group()

        if file_to_request not in existing_files:
            try:
                location = os.getcwd() + dataset + category + '/' + file_to_request
                image_file = open(location, 'x')
                urllib.request.urlretrieve(url, location)
                if os.stat(location).st_size == 0:
                    os.remove(location)
            except FileExistsError:
                continue
        else:
            print("Image file " + '\'' + file_to_request + '\'' + " already exists.")

