from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Uncomment to run in headless mode
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--allow-insecure-localhost")

# Initialize WebDriver
driver = webdriver.Chrome(options=chrome_options)

try:
    # Navigate to your website
    driver.get('http://deei-mooshak.ualg.pt/dlchan/')  # Replace with your actual URL

    # Optional: Wait to observe the page
    time.sleep(5)

    # Wait for the Dropzone form to load
    dropzone = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'droparea'))
    )
    print("Dropzone element found:", dropzone)

    # Try different methods to locate the file input element
    # Method 1: Using CSS Selector with class
    try:
        file_input = driver.find_element(By.CSS_SELECTOR, 'input.dz-hidden-input')
    except:
        # Method 2: Using XPath
        file_input = driver.find_element(By.XPATH, '//input[@type="file"]')

    # Optionally, make the file input visible
    driver.execute_script("arguments[0].style.display = 'block';", file_input)

    # Specify the path to your file
    file_path = 'C:\\Users\\diogo\\Documents\\GitHub\\AM\\T1-KNearestNeighbors.pickle'  # Replace with your file's path

    # Send the file path to the input element
    file_input.send_keys(file_path)
    print("File path sent to input.")

    # Wait for the upload to complete
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'dz-success'))
    )
    print("File uploaded successfully.")


finally:
    # Close the browser
    driver.quit()
