import os
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, MoveTargetOutOfBoundsException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from urllib.parse import urljoin
import time

def load_page_with_retry(driver, url, max_retries=1):
    """
    Attempts to load a webpage with retry logic in case of failures.
    
    Args:
        driver: Selenium WebDriver instance
        url: URL to load
        max_retries: Maximum number of retry attempts (default: 1)
    
    Returns:
        bool: True if page loaded successfully, False otherwise
    """
    for attempt in range(max_retries):
        try:
            print(f"Попытка загрузки {attempt + 1}/{max_retries}...")
            driver.get(url)
            time.sleep(random.uniform(2, 5))
            return True
        except TimeoutException:
            print(f"Таймаут при загрузке, попытка {attempt + 1}")
            if attempt == max_retries - 1:
                return False
            setup_environment()
            time.sleep(random.randint(3, 6))
    return False


def setup_environment():
    """
    Cleans up the system environment by terminating Chrome processes.
    Works for both Windows (nt) and Unix-like systems.
    """
    try:
        if os.name == 'nt':
            os.system("taskkill /f /im chrome.exe /t")
            os.system("taskkill /f /im chromedriver.exe /t")
        else:
            os.system("pkill -f chrome")
            os.system("pkill -f chromedriver")
        time.sleep(1)
    except:
        pass


def create_driver():
    """
    Creates and configures a Chrome WebDriver instance with optimized settings.
    
    Returns:
        WebDriver: Configured Chrome WebDriver instance
    """
    chrome_options = webdriver.ChromeOptions()
    
    # Core configuration
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    
    # Performance optimizations
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--start-maximized")
    
    # Настройки User-Agent и языка
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.35")
    chrome_options.add_argument("--lang=en-US,en;q=0.9")
    
    # User-agent and language settings
    service = Service(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Initialize ChromeDriver service
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """
    })
    
    # Mask WebDriver detection
    driver.set_page_load_timeout(60)
    driver.implicitly_wait(20)
    
    return driver


def extract_project_media(driver, base_url):
    """
    Extracts media content (images and videos) from a Kickstarter project page.
    
    Args:
        driver: Selenium WebDriver instance
        base_url: Base URL of the project for resolving relative URLs
    
    Returns:
        dict: Dictionary containing lists of image and video URLs
    """
    
    media = {
        'images': [],
        'videos': []
    }
    description_container_selector = "div.rte__content"
    
    # 1. Extract main video content
    try:
        # Kickstarter video player detection
        video_sources = driver.find_elements(
            By.CSS_SELECTOR, 
            "div.ksr-video-player source[src*='.mp4'], "
            "div.video-player source[src*='.mp4']"
        )
        
        # Collect unique MP4 URLs from kickstarter.com
        unique_videos = set()
        for source in video_sources:
            src = source.get_attribute('src')
            if src and 'kickstarter.com' in src:
                unique_videos.add(src.split('?')[0])  # Remove URL parameters
        
        media['videos'] = list(unique_videos)
        
        # Fallback to YouTube detection if no Kickstarter videos found
        if not media['videos']:
            yt_link = description_container.find_element(
                By.CSS_SELECTOR, 
                "div.clip"
            )
            if yt_link:
                yt_url = yt_link.get_attribute('href').split('&')[0]
                if '/watch?v=' in yt_url:
                    media['videos'] = [yt_url]
    
    except Exception:
        pass

    # 2. Extract image content
    try:
        description_container = driver.find_element(By.CSS_SELECTOR, description_container_selector)
        img_elements = description_container.find_elements(
            By.CSS_SELECTOR,
            "img[src], img[data-src], [data-image]"
        )
        
        for img in img_elements:
            src = (img.get_attribute('src') or 
                  img.get_attribute('data-src') or
                  img.get_attribute('data-image'))
            
            if src and not src.startswith(('data:', 'blob:')):
                if not src.startswith(('http://', 'https://')):
                    src = urljoin(base_url, src)
                if src not in media['images']:
                    media['images'].append(src)
    except Exception as e:
        print(f"Ошибка парсинга изображений: {str(e)}")

    return media


def parse_page(driver, url):
    """
    Parses a Kickstarter project page for description and media content.
    
    Args:
        driver: Selenium WebDriver instance
        url: URL of the project page
    
    Returns:
        dict: Dictionary containing parsed description and media content
    """
    result = {
        'description': '',
        'media': {
            'images': [],
            'videos': []
        }
    }
    
    try:
        print("Ожидание контента проекта...")
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'rte__content') or contains(@class, 'story-content')]"))
        )
        
        # Try multiple selectors for description extraction
        selectors = [
            (By.CSS_SELECTOR, "div.rte__content"),
        ]
        
        for by, selector in selectors:
            try:
                element = WebDriverWait(driver, 15).until(
                    EC.visibility_of_element_located((by, selector)))
                content = element.text.strip()
                if content:
                    print(f"Описание найдено с помощью {by} и селектора {selector}!")
                    result['description'] = content
                    break
            except:
                continue
        
        if not result['description']:
            result['description'] = "Контент не найден"
        
        # Extract media content
        print("Извлечение медиа-контента проекта...")
        result['media'] = extract_project_media(driver, url)
        
        return result
    
    except Exception as e:
        print(f"Ошибка парсинга: {str(e)}")
        driver.save_screenshot("error.png")
        result['description'] = f"Ошибка: {str(e)}"
        return result


def save_results(result, base_filename="project_result"):
    """
    Saves parsed results to text files in a structured format.
    
    Args:
        result: Dictionary containing parsed data
        base_filename: Base name for output files
    """
    # Save description text
    with open(f"kickstarter_data/{base_filename}_description.txt", "w", encoding="utf-8") as f:
        f.write(result['description'])
        
    # Save image URLs
    with open(f"kickstarter_data/{base_filename}_images.txt", "w", encoding="utf-8") as f:
        for img in result['media']['images']:
            f.write(img + "\n")
    
    # Save video URLs
    with open(f"kickstarter_data/{base_filename}_videos.txt", "w", encoding="utf-8") as f:
        for video in result['media']['videos']:
            f.write(video + "\n")
    
    print(f"\nСохранено {len(result['media']['images'])} изображений и {len(result['media']['videos'])} видео")

def get_description(url):
    """
    Main function to coordinate the scraping process.
    
    Args:
        url: Kickstarter project URL to scrape
    
    Returns:
        dict: Parsed project data
    """
    setup_environment()
    driver = None
    try:
        driver = create_driver()
        
        print("\nНачало работы парсера...")
        
        driver.get(url)

        result = parse_page(driver, url)
        
        print("\nРезультат парсинга:")
        print(f"Описание: {result['description'][:200]}...")
        print(f"Найдено изображений проекта: {len(result['media']['images'])}")
        print(f"Найдено видео проекта: {len(result['media']['videos'])}")
        
        save_results(result)

        
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
    finally:
        if driver:
            driver.quit()
        print("\nРабота завершена")
          
    return result
    
if __name__ == "__main__":
    # Example usage
    url = "https://www.kickstarter.com/projects/futurecatgames/margin-of-the-strange" #Test URL
    start_time = time.time()
    get_description(url)
    end_time = time.time()
    print("Затраченное время на проект:", end_time - start_time) 
    
