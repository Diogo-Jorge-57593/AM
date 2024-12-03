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
    file_path = 'C:\\Users\\diogo\\Documents\\GitHub\\AM\\T1-SVM.pickle'  # Replace with your file's path

    # Send the file path to the input element
    file_input.send_keys(file_path)
    print("File path sent to input.")

    # Wait for the upload to complete
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, 'dz-success'))
    )
    print("File uploaded successfully.")

    # Capture live accuracy during model execution
    live_accuracy = None
    previous_accuracy = 0.0
    no_improvement_count = 0  # Tracks how long accuracy has not improved
    max_no_improvement = 5  # Number of checks without improvement before stopping

    print("Waiting for live accuracy to stabilize...")
    while True:
        try:
            # Locate the live accuracy element
            accuracy_element = WebDriverWait(driver, 2).until(
                EC.presence_of_element_located((By.ID, 'acc'))
            )
            live_accuracy = float(accuracy_element.text.strip())
            print(f"Current Live Accuracy: {live_accuracy:.5f}")

            # Check if accuracy improved
            if live_accuracy > previous_accuracy:
                previous_accuracy = live_accuracy
                no_improvement_count = 0  # Reset counter if there is improvement
            else:
                no_improvement_count += 1
                print(f"No improvement detected. Count: {no_improvement_count}")

            # Stop if no improvement for a while
            if no_improvement_count >= max_no_improvement:
                print("Accuracy stabilized or no further improvement detected.")
                break

            # Add a short delay before re-checking
            time.sleep(1)
        except Exception as e:
            print(f"Error while checking live accuracy: {e}")
            break

    # Save the final accuracy to a log file
    log_data = {
        "final_live_accuracy": previous_accuracy,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    log_file = "live_accuracy_log.json"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_data) + "\n")
    print(f"Final live accuracy logged to {log_file}.")

finally:
    # Close the browser
    driver.quit()
