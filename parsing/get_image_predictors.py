import pandas as pd
import numpy as np
import requests
from PIL import Image, ImageStat
from io import BytesIO
import cv2
import ast
#from tqdm import tqdm
import warnings
import re
from urllib.parse import urlparse, parse_qs
import os

warnings.filterwarnings('ignore')

def get_image_characteristics(image_url):
    """
    Analyzes an image from a URL and extracts various visual characteristics.
    
    Args:
        image_url (str): URL of the image to analyze
        
    Returns:
        dict: Dictionary containing image metrics or None if analysis fails
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.35',
            'Referer': 'https://www.kickstarter.com/',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
        }

        response = requests.get(image_url, headers=headers, stream=True, timeout=10)
        response.raise_for_status()

        if 'image' not in response.headers.get('Content-Type', ''):
            return None

        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img)
        gray_img = img.convert('L')
        stat = ImageStat.Stat(img)
        stat_gray = ImageStat.Stat(gray_img)

        contrast = stat_gray.rms[0] if hasattr(stat_gray, 'rms') else np.nan

        return {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode,
            'file_size_kb': len(response.content) / 1024,
            'brightness_mean': stat_gray.mean[0] if hasattr(stat_gray, 'mean') else np.nan,
            'brightness_std': stat_gray.stddev[0] if hasattr(stat_gray, 'stddev') else np.nan,
            'contrast_rms': contrast,
            'r_mean': stat.mean[0] if len(stat.mean) > 0 else np.nan,
            'g_mean': stat.mean[1] if len(stat.mean) > 1 else np.nan,
            'b_mean': stat.mean[2] if len(stat.mean) > 2 else np.nan,
            'entropy': calculate_entropy(gray_img),
            'sharpness': calculate_sharpness(img_array),
            'hsv_brightness': calculate_brightness_hsv(img_array),
            'saturation': calculate_saturation(img_array),
            'aspect_ratio': img.width / img.height,
            'unique_colors': count_unique_colors(img)
        }
    except Exception as e:
        print(f"Ошибка при обработке {image_url}: {str(e)}")
        return None

def calculate_entropy(image):
    """
    Calculates the Shannon entropy of an image (measure of information content).
    
    Args:
        image: PIL Image object (grayscale)
        
    Returns:
        float: Entropy value
    """
    hist = np.array(image.histogram())
    hist = hist / hist.sum()
    hist = hist[hist != 0]
    return -np.sum(hist * np.log2(hist))

def calculate_sharpness(image_array):
    """
    Estimates image sharpness using Laplacian variance.
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        float: Sharpness metric
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_brightness_hsv(image_array):
    """
    Calculates brightness in HSV color space (V channel).
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        float: Mean brightness value
    """
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    return np.mean(hsv[:,:,2])

def calculate_saturation(image_array):
    """
    Calculates color saturation in HSV color space (S channel).
    
    Args:
        image_array: Numpy array of the image
        
    Returns:
        float: Mean saturation value
    """
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    return np.mean(hsv[:,:,1])

def count_unique_colors(image):
    """
    Counts the number of unique colors in an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        int: Number of unique colors
    """
    try:
        colors = image.getcolors(maxcolors=2**24)
        return len(colors) if colors else 0
    except:
        return 0

def analyze_image_url(url):
    """
    Extracts metadata features from image URL.
    
    Args:
        url (str): Image URL to analyze
        
    Returns:
        dict: Dictionary of URL-based features
    """
    if not isinstance(url, str):
        return {}

    features = {}

    # 1. Extract file extension
    extension_match = re.search(r'\.([a-zA-Z0-9]+)(\?|$)', url)
    if extension_match:
        features['extension'] = extension_match.group(1).lower()
    else:
        features['extension'] = 'unknown'

    # 2. Parse URL query parameters
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        if 'q' in query_params:
            features['quality'] = int(query_params['q'][0])
        else:
            features['quality'] = None

        if 'width' in query_params:
            features['width'] = int(query_params['width'][0])
        else:
            features['width'] = None

        if 'v' in query_params:
            features['timestamp'] = int(query_params['v'][0])
        else:
            features['timestamp'] = None

        if 'anim' in query_params:
            features['animated'] = query_params['anim'][0].lower() == 'true'
        else:
            features['animated'] = features['extension'] == 'gif'

    except:
        features['quality'] = None
        features['width'] = None
        features['timestamp'] = None
        features['animated'] = features['extension'] == 'gif'

    return features

def extract_image_features(urls):
    """
    Analyzes a list of image URLs and extracts aggregate features.
    
    Args:
        urls (list): List of image URLs
        
    Returns:
        dict: Dictionary of aggregate image features
    """
    if not urls:
        return {
            'has_gif': False,
            'has_png': False,
            'has_jpg': False,
            'has_animated_gif': False
        }

    image_features = [analyze_image_url(url) for url in urls]

    features = {}

    extensions = [f.get('extension', 'unknown') for f in image_features]
    features['has_gif'] = 'gif' in extensions
    features['has_png'] = 'png' in extensions
    features['has_jpg'] = any(ext in ['jpg', 'jpeg'] for ext in extensions)
    
    features['has_animated_gif'] = any(f.get('animated', False) for f in image_features)

    return features

def add_image_features_to_dataset(df, images_column='images', output_file='enhanced_dataset.csv', force_update=False):
    """
    Enhances a DataFrame with image analysis features.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        images_column (str): Name of column containing image URLs
        output_file (str): Path to save enhanced dataset
        force_update (bool): Whether to reprocess existing records
        
    Returns:
        pd.DataFrame: Enhanced DataFrame
    """
    if os.path.exists(output_file) and not force_update:
        existing_df = pd.read_csv(output_file)
        existing_ids = set(existing_df['id']) if 'id' in existing_df.columns else set()
    else:
        existing_df = pd.DataFrame()
        existing_ids = set()

    image_features = [
        'img_count',
        'avg_width', 'avg_height', 'avg_aspect_ratio',
        'avg_brightness', 'max_brightness', 'min_brightness',
        'avg_contrast', 'max_contrast', 'min_contrast',
        'avg_sharpness', 'max_sharpness', 'min_sharpness',
        'avg_saturation', 'max_saturation', 'min_saturation',
        'avg_entropy', 'max_entropy', 'min_entropy',
        'total_unique_colors', 'avg_unique_colors',
        'has_images',
        'has_gif',
        'has_png',
        'has_jpg', 
        'has_animated_gif'
    ]

    for col in image_features:
        if col not in df.columns:
            df[col] = np.nan

    if 'id' in df.columns:
        rows_to_process = df[~df['id'].isin(existing_ids)].iterrows()
    else:
        rows_to_process = df.iterrows()

    pbar = tqdm(rows_to_process, total=len(df) if 'id' not in df.columns else len(df[~df['id'].isin(existing_ids)]))
    
    for idx, row in pbar:
        try:
            if 'id' in row and row['id'] in existing_ids:
                continue

            if isinstance(row[images_column], str):
                urls = ast.literal_eval(row[images_column])
            else:
                urls = row[images_column]

            if not urls:
                df.at[idx, 'has_images'] = False
                continue

            url_features = extract_image_features(urls)
            for feature, value in url_features.items():
                df.at[idx, feature] = value

            df.at[idx, 'has_images'] = len(urls) > 0

            all_features = []
            valid_images = 0

            for url in urls:
                features = get_image_characteristics(url)
                if features:
                    all_features.append(features)
                    valid_images += 1

            if not all_features:
                continue

            features_df = pd.DataFrame(all_features)

            features_df.fillna({
                'brightness_mean': features_df['brightness_mean'].mean(),
                'contrast_rms': features_df['contrast_rms'].mean(),
                'sharpness': features_df['sharpness'].mean(),
                'saturation': features_df['saturation'].mean(),
                'entropy': features_df['entropy'].mean()
            }, inplace=True)

            df.at[idx, 'img_count'] = valid_images
            df.at[idx, 'avg_width'] = features_df['width'].mean()
            df.at[idx, 'avg_height'] = features_df['height'].mean()
            df.at[idx, 'avg_aspect_ratio'] = features_df['aspect_ratio'].mean()

            for prop in ['brightness_mean', 'contrast_rms', 'sharpness', 'saturation', 'entropy']:
                col_prefix = prop.replace('_mean', '') if '_mean' in prop else prop
                df.at[idx, f'avg_{col_prefix}'] = features_df[prop].mean()
                df.at[idx, f'max_{col_prefix}'] = features_df[prop].max()
                df.at[idx, f'min_{col_prefix}'] = features_df[prop].min()

            df.at[idx, 'total_unique_colors'] = features_df['unique_colors'].sum()
            df.at[idx, 'avg_unique_colors'] = features_df['unique_colors'].mean()

            if 'id' in df.columns:
                if os.path.exists(output_file):
                    existing_df = pd.read_csv(output_file)
                    updated_df = pd.concat([existing_df, df.loc[[idx]]], ignore_index=True)
                else:
                    updated_df = df.loc[[idx]]
                
                if 'id' in updated_df.columns:
                    updated_df = updated_df.drop_duplicates(subset='id', keep='last')
                
                updated_df.to_csv(output_file, index=False)

        except Exception as e:
            print(f"\nОшибка в строке {idx}: {str(e)}")
            continue

    return df

if __name__ == "__main__":
    """
    Main workflow:
    1. Loads source data file
    2. Cleans data (removes duplicates and invalid entries)
    3. Enhances data with image features
    4. Saves enhanced dataset
    """
    input_file = 'combined_df.csv'  
    output_file = 'enhanced_dataset.csv'
    
    df = pd.read_csv(input_file)
    
    df = df.drop_duplicates(subset='name')
    print(f"Размер датасета после удаления дубликатов: {df.shape}")

    df = df[~df['description'].str.contains("Контент не найден|Ошибка: Message:", na=False)]
    print(f"Размер датасета после удаления строк с ошибками: {df.shape}")

    print("\nСтруктура целевой переменной (state):")
    print(df['state'].value_counts())
    
    df['images'] = df['images'].apply(ast.literal_eval)

    enhanced_df = add_image_features_to_dataset(df, images_column='images', output_file=output_file)

    print(f"\nДатасет с новыми признаками сохранен как '{output_file}'")