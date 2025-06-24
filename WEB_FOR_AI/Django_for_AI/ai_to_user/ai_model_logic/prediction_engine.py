import pandas as pd
import numpy as np
import joblib
from datetime import datetime, time, timedelta
import requests
from PIL import Image, ImageStat
from io import BytesIO
import cv2
import re
from urllib.parse import urlparse, parse_qs

# Функции для обработки изображений (остаются без изменений)
def get_image_characteristics(image_url):
    """Анализирует изображение по URL и извлекает визуальные характеристики."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.kickstarter.com/'
        }

        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.array(img)
        gray_img = img.convert('L')
        stat = ImageStat.Stat(img)
        stat_gray = ImageStat.Stat(gray_img)

        return {
            'width': img.width,
            'height': img.height,
            'aspect_ratio': img.width / img.height,
            'brightness': stat_gray.mean[0],
            'contrast_rms': stat_gray.rms[0],
            'sharpness': calculate_sharpness(img_array),
            'saturation': calculate_saturation(img_array),
            'entropy': calculate_entropy(gray_img),
            'unique_colors': count_unique_colors(img)
        }
    except Exception as e:
        print(f"Ошибка при обработке изображения: {str(e)}")
        return None

def calculate_entropy(image):
    """Вычисляет энтропию изображения."""
    hist = np.array(image.histogram())
    hist = hist / hist.sum()
    hist = hist[hist != 0]
    return -np.sum(hist * np.log2(hist))

def calculate_sharpness(image_array):
    """Оценивает резкость изображения."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_saturation(image_array):
    """Вычисляет насыщенность цветов."""
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    return np.mean(hsv[:,:,1])

def count_unique_colors(image):
    """Подсчитывает количество уникальных цветов."""
    try:
        colors = image.getcolors(maxcolors=2**24)
        return len(colors) if colors else 0
    except:
        return 0

def analyze_image(url):
    """Анализирует изображение и возвращает признаки."""
    if not url:
        return None
    
    # Сначала проверяем URL
    url_features = {
        'has_image': 1,
        'is_gif': 0,
        'is_png': 0,
        'is_jpg': 0
    }
    
    ext = url.split('.')[-1].lower().split('?')[0]
    if ext == 'gif':
        url_features['is_gif'] = 1
    elif ext == 'png':
        url_features['is_png'] = 1
    elif ext in ['jpg', 'jpeg']:
        url_features['is_jpg'] = 1
    
    # Затем анализируем само изображение
    img_features = get_image_characteristics(url)
    if img_features:
        url_features.update(img_features)
    
    return url_features

# Загрузка модели
try:
    # Указываем полный относительный путь от корня проекта Django
    model_path = 'ai_to_user/ai_model_logic/kickstarter_success/classification/boosting/optimized_xgboost_model.joblib'
    model = joblib.load(model_path)

    # Если у вас есть другие файлы (энкодеры, список признаков),
    # укажите их пути аналогичным образом:
    # country_encoder = joblib.load('ai_to_user/ai_model_logic/country_encoder.pkl')
    # onehot_encoder = joblib.load('ai_to_user/ai_model_logic/onehot_encoder.pkl')
    # model_features = joblib.load('ai_to_user/ai_model_logic/model_features.pkl')

    print("Модель и необходимые компоненты успешно загружены.")
    # Убедитесь, что все переменные (model, country_encoder и т.д.)
    # инициализированы здесь, если они используются в predict_success.
except FileNotFoundError as e:
    print(f"Ошибка загрузки модели или энкодеров: Файл не найден по пути {e.filename}. Убедитесь, что файлы существуют по указанным путям.")
    model = None
    # Инициализируйте остальные компоненты как None или пустые структуры, если они не загрузились
    # country_encoder = None
    # onehot_encoder = None
    # model_features = []
except Exception as e:
    print(f"Неожиданная ошибка при загрузке модели или энкодеров: {e}")
    model = None
    # Аналогично для других компонентов

# Функция для преобразования времени
def _get_time_bin(time_obj):
    """Функция определения временного интервала с правильными названиями бинов."""
    if isinstance(time_obj, str):
        try:
            try:
                time_obj = datetime.strptime(time_obj, '%H:%M:%S').time()
            except:
                time_obj = datetime.strptime(time_obj, '%H:%M').time()
        except:
            return '10am-12pm'  # Значение по умолчанию
    elif not isinstance(time_obj, time):
        return '10am-12pm'
    
    hour = time_obj.hour
    
    # Используем точные названия бинов, которые есть в expected_features
    if 0 <= hour < 2: return '12am-2am'
    elif 2 <= hour < 4: return '2am-4am'
    elif 4 <= hour < 6: return '4am-6am'
    elif 6 <= hour < 8: return '6am-8am'
    elif 8 <= hour < 10: return '8am-10am'
    elif 10 <= hour < 12: return '10am-12pm'
    elif 12 <= hour < 14: return '12pm-2pm'
    elif 14 <= hour < 16: return '2pm-4pm'
    elif 16 <= hour < 18: return '4pm-6pm'  # Для 16:30
    elif 18 <= hour < 20: return '6pm-8pm'
    elif 20 <= hour < 22: return '8pm-10pm'
    elif 22 <= hour <= 23: return '10pm-12am'
    return '10am-12pm'

def predict_success(raw_data):
    """Основная функция для предсказания успеха проекта."""
    if model is None:
        return "Ошибка: Модель не загружена", None
    
    # 1. Подготовка базовых признаков
    features = {
        'name_length': len(raw_data.get('name', '')),
        'blurb_length': len(raw_data.get('blurb', '')),
        'description_length': len(raw_data.get('description', '')),
        'usd_goal': float(raw_data.get('usd_goal', 0)),
        'campaign_days': int(raw_data.get('campaign_days', 0)),
        'prelaunch_activated': 1 if raw_data.get('prelaunch_activated', False) else 0,
        'staff_pick': 0,
        'is_liked': 0,
        'image_count': 0,
        'video_count': 1 if raw_data.get('video') else 0,
        'has_animated_gif': 0,
        'img_count': 0,
    }

    # 2. Обработка изображений (теперь принимаем список изображений)
    image_urls = raw_data.get('project_image', [])
    if not isinstance(image_urls, list):
        image_urls = [image_urls] if image_urls else []
    
    img_features_list = []
    for image_url in image_urls:
        if image_url:
            img_features = analyze_image(image_url)
            if img_features:
                img_features_list.append(img_features)
    
    if img_features_list:
        # Рассчитываем средние значения по всем изображениям
        avg_features = {
            'image_count': len(img_features_list),
            'img_count': len(img_features_list),
            'has_images': 1,
            'has_gif': sum(f.get('is_gif', 0) for f in img_features_list),
            'has_png': sum(f.get('is_png', 0) for f in img_features_list),
            'has_jpg': sum(f.get('is_jpg', 0) for f in img_features_list),
            'avg_width': np.mean([f.get('width', 0) for f in img_features_list]),
            'avg_height': np.mean([f.get('height', 0) for f in img_features_list]),
            'avg_aspect_ratio': np.mean([f.get('aspect_ratio', 0) for f in img_features_list]),
            'avg_brightness': np.mean([f.get('brightness', 0) for f in img_features_list]),
            'max_brightness': np.max([f.get('brightness', 0) for f in img_features_list]),
            'min_brightness': np.min([f.get('brightness', 0) for f in img_features_list]),
            'avg_sharpness': np.mean([f.get('sharpness', 0) for f in img_features_list]),
            'max_sharpness': np.max([f.get('sharpness', 0) for f in img_features_list]),
            'min_sharpness': np.min([f.get('sharpness', 0) for f in img_features_list]),
            'avg_saturation': np.mean([f.get('saturation', 0) for f in img_features_list]),
            'max_saturation': np.max([f.get('saturation', 0) for f in img_features_list]),
            'min_saturation': np.min([f.get('saturation', 0) for f in img_features_list]),
            'avg_entropy': np.mean([f.get('entropy', 0) for f in img_features_list]),
            'max_entropy': np.max([f.get('entropy', 0) for f in img_features_list]),
            'min_entropy': np.min([f.get('entropy', 0) for f in img_features_list]),
            'avg_contrast_rms': np.mean([f.get('contrast_rms', 0) for f in img_features_list]),
            'max_contrast_rms': np.max([f.get('contrast_rms', 0) for f in img_features_list]),
            'min_contrast_rms': np.min([f.get('contrast_rms', 0) for f in img_features_list]),
            'total_unique_colors': sum(f.get('unique_colors', 0) for f in img_features_list),
            'avg_unique_colors': np.mean([f.get('unique_colors', 0) for f in img_features_list]),
        }
        features.update(avg_features)

    # 3. Обработка дат и времени
    creation_date = raw_data.get('creation_date')
    launch_date = raw_data.get('launch_date')
    
    # Временные интервалы (используем формат как в модели)
    launch_time = raw_data.get('launch_time', time(10, 0))  # Значение по умолчанию 10:00
    launch_time_bin = _get_time_bin(launch_time)

    # Сбрасываем все временные бины
    time_bins = [
        '10am-12pm', '10pm-12am', '12am-2am', '12pm-2pm',
        '2am-4am', '2pm-4pm', '4am-6am', '4pm-6pm',
        '6am-8am', '6pm-8pm', '8am-10am', '8pm-10pm'
        ]

    # Сбрасываем все временные бины
    for time_bin in time_bins:
        features[f'launch_time_{time_bin}'] = 0
        features[f'deadline_time_{time_bin}'] = 0

    # Устанавливаем правильные бины
    launch_time_bin = _get_time_bin(raw_data.get('launch_time', time(10, 0)))
    features[f'launch_time_{launch_time_bin}'] = 1
    features[f'deadline_time_{launch_time_bin}'] = 1
    
    # Дни недели и месяцы (используем правильный регистр как в модели)
    if creation_date:
        created_day = creation_date.strftime('%A')
        created_month = creation_date.strftime('%B')
        features[f'created_day_{created_day}'] = 1
        features[f'created_month_{created_month}'] = 1

    if launch_date:
        launch_day = launch_date.strftime('%A')
        launch_month = launch_date.strftime('%B')
        features[f'launch_day_{launch_day}'] = 1
        features[f'launch_month_{launch_month}'] = 1

        # Расчет deadline
        deadline_date = launch_date + timedelta(days=features['campaign_days'])
        deadline_day = deadline_date.strftime('%A')
        deadline_month = deadline_date.strftime('%B')
        deadline_time_bin = _get_time_bin(raw_data.get('launch_time', '10:00:00'))
        
        features[f'deadline_day_{deadline_day}'] = 1
        features[f'deadline_month_{deadline_month}'] = 1
        features[f'deadline_time_{deadline_time_bin}'] = 1

    # Дни между созданием и запуском
    if creation_date and launch_date:
        features['creation_to_launch_days'] = (launch_date - creation_date).days
    else:
        features['creation_to_launch_days'] = 0

    # 4. Обработка страны (используем формат как в модели)
    country = raw_data.get('country', 'The United States').strip()  # Получаем и очищаем строку
    
    # Словарь для преобразования введенных стран в формат модели
    country_mapping = {
        'the united states': 'the united states',
        'the united kingdom': 'the united kingdom',
        'the netherlands': 'the netherlands',
        'hong kong': 'hong kong',
        'new zealand': 'new zealand',
        'russia': 'russia',
        'rus': 'russia',
        'poland': 'poland',
        'italy': 'italy',
        'germany': 'germany',
        'france': 'france',
        'spain': 'spain',
        'canada': 'canada',
        'australia': 'australia',
        'japan': 'japan',
        'mexico': 'mexico',
        'switzerland': 'switzerland',
        'sweden': 'sweden',
        'norway': 'norway',
        'denmark': 'denmark',
        'austria': 'austria',
        'belgium': 'belgium',
        'greece': 'greece',
        'ireland': 'ireland',
        'luxembourg': 'luxembourg',
        'singapore': 'singapore',
        'slovenia': 'slovenia'
    }
    
    # Приводим к нижнему регистру для сравнения
    country_lower = country.lower()
    
    # Ищем страну в маппинге или оставляем как есть (в нижнем регистре)
    model_country = country_mapping.get(country_lower, country_lower)
    
    # Удаляем все существующие country_ признаки
    features = {k: v for k, v in features.items() if not k.startswith('country_')}
    
    # Устанавливаем правильный признак страны
    features[f'country_{model_country}'] = 1

    # 5. Создание финального DataFrame с правильным порядком признаков
    expected_features = [
        'is_liked', 'prelaunch_activated', 'staff_pick', 'img_count',
        'avg_width', 'avg_height', 'avg_aspect_ratio', 'avg_brightness',
        'max_brightness', 'min_brightness', 'avg_sharpness', 'max_sharpness',
        'min_sharpness', 'avg_saturation', 'max_saturation', 'min_saturation',
        'avg_entropy', 'max_entropy', 'min_entropy', 'total_unique_colors',
        'avg_unique_colors', 'has_images', 'has_gif', 'has_png', 'has_jpg',
        'has_animated_gif', 'avg_contrast_rms', 'max_contrast_rms',
        'min_contrast_rms', 'image_count', 'video_count', 'description_length',
        'blurb_length', 'usd_goal', 'name_length', 'creation_to_launch_days',
        'campaign_days', 'deadline_day_Friday', 'deadline_day_Monday',
        'deadline_day_Saturday', 'deadline_day_Sunday', 'deadline_day_Thursday',
        'deadline_day_Tuesday', 'deadline_day_Wednesday', 'launch_time_10am-12pm',
        'launch_time_10pm-12am', 'launch_time_12am-2am', 'launch_time_12pm-2pm',
        'launch_time_2am-4am', 'launch_time_2pm-4pm', 'launch_time_4am-6am',
        'launch_time_4pm-6pm', 'launch_time_6am-8am', 'launch_time_6pm-8pm',
        'launch_time_8am-10am', 'launch_time_8pm-10pm', 'created_month_April',
        'created_month_August', 'created_month_December', 'created_month_February',
        'created_month_January', 'created_month_July', 'created_month_June',
        'created_month_March', 'created_month_May', 'created_month_November',
        'created_month_October', 'created_month_September', 'launch_month_April',
        'launch_month_August', 'launch_month_December', 'launch_month_February',
        'launch_month_January', 'launch_month_July', 'launch_month_June',
        'launch_month_March', 'launch_month_May', 'launch_month_November',
        'launch_month_October', 'launch_month_September', 'deadline_time_10am-12pm',
        'deadline_time_10pm-12am', 'deadline_time_12am-2am', 'deadline_time_12pm-2pm',
        'deadline_time_2am-4am', 'deadline_time_2pm-4pm', 'deadline_time_4am-6am',
        'deadline_time_4pm-6pm', 'deadline_time_6am-8am', 'deadline_time_6pm-8pm',
        'deadline_time_8am-10am', 'deadline_time_8pm-10pm', 'created_day_Friday',
        'created_day_Monday', 'created_day_Saturday', 'created_day_Sunday',
        'created_day_Thursday', 'created_day_Tuesday', 'created_day_Wednesday',
        'country_australia', 'country_austria', 'country_belgium', 'country_canada',
        'country_denmark', 'country_france', 'country_germany', 'country_greece',
        'country_hong kong', 'country_ireland', 'country_italy', 'country_japan',
        'country_luxembourg', 'country_mexico', 'country_new zealand',
        'country_norway', 'country_poland', 'country_singapore', 'country_slovenia',
        'country_spain', 'country_sweden', 'country_switzerland',
        'country_the netherlands', 'country_the united kingdom',
        'country_the united states', 'deadline_month_April', 'deadline_month_August',
        'deadline_month_December', 'deadline_month_February', 'deadline_month_January',
        'deadline_month_July', 'deadline_month_June', 'deadline_month_March',
        'deadline_month_May', 'deadline_month_November', 'deadline_month_October',
        'deadline_month_September', 'launch_day_Friday', 'launch_day_Monday',
        'launch_day_Saturday', 'launch_day_Sunday', 'launch_day_Thursday',
        'launch_day_Tuesday', 'launch_day_Wednesday'
    ]

    # Создаем DataFrame и заполняем недостающие признаки нулями
    final_df = pd.DataFrame([features])
    
    # Добавляем все ожидаемые признаки
    for col in expected_features:
        if col not in final_df.columns:
            final_df[col] = 0
    
    # Убедимся, что порядок колонок правильный
    final_df = final_df[expected_features]

    # 6. Предсказание
    try:
        prediction_proba = model.predict_proba(final_df)[0][1]
        prediction_label = model.predict(final_df)[0]
        
        result = f"Проект будет успешным! (Вероятность: {prediction_proba:.2%})" if prediction_label == 1 else \
                f"Проект может провалиться. (Вероятность успеха: {prediction_proba:.2%})"
        return result, final_df
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return "Ошибка при выполнении предсказания.", None

# Пример использования с тестовыми данными
if __name__ == "__main__":
    from datetime import date, time
    
    test_data = {
        'name': 'Проект Марины',
        'blurb': 'Меня зовут Марина',
        'country': 'Hong Kong',
        'usd_goal': 8000.0,
        'campaign_days': 25,
        'prelaunch_activated': False,
        'creation_date': date(2025, 6, 7),
        'creation_time': time(16, 30),
        'launch_date': date(2025, 6, 7),
        'launch_time': time(16, 30),
        'description': 'маринкин проект для самых прикольных чуваков йоооооуууууууу кто пропустил тот лох',
        'project_image': ['https://i.kickstarter.com/assets/048/781/840/313fa61403c863b33a231788ff116b1b_original.jpg?fit=scale-down&origin=ugc&q=92&v=1743753100&width=680&sig=F6ugshAneIT4DiGHAtFNyHlX7kK4NEcy0kEWgE0ZBD8%3D','https://i.kickstarter.com/assets/049/441/616/9d4b5966bb52add45b6562dd20718e16_original.png?fit=scale-down&origin=ugc&q=100&v=1748363583&width=680&sig=HKH2TB%2F%2BOOM2VL9mP1edbYoc7ON6%2BW27omj2AkpBqe4%3D'],
        'video': 'https://www.youtube.com/watch?v=mN2VLo-74BE'
    }
    
    result, final_df = predict_success(test_data)
    print(result)
    
    if final_df is not None:
        # Дополнительный вывод: показываем финальные данные, передаваемые в модель
        print("\nИтоговые данные, передаваемые в модель:")
        
        # Получаем первую строку DataFrame
        final_features = final_df.iloc[0]
        
        for feature, value in final_features.items():
            print(f"{feature}: {value}")