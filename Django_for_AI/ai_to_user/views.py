# ai_to_user/views.py  

from django.shortcuts import render, redirect  
from django.contrib import messages  
from .forms import ProjectPredictionForm  
from .models import Project  

# Импортируем функцию predict_success из вашего модуля  
from .ai_model_logic.prediction_engine import predict_success  

# --- Ваша view-функция: predict_view ---  
def predict_view(request):  
    ai_prediction_result = None  

    if request.method == 'POST':  
        form = ProjectPredictionForm(request.POST, request.FILES)  
        if form.is_valid():  
            cleaned_data = form.cleaned_data  

            # --- ОБРАБОТКА МНОЖЕСТВЕННЫХ ССЫЛОК НА ИЗОБРАЖЕНИЯ ---
            # Получаем строку со ссылками
            image_urls_str = cleaned_data.get('project_image', '')
            # Разделяем строку по запятым или символу новой строки и удаляем лишние пробелы
            image_urls_list = []
            if image_urls_str:
                # Сначала разделим по новой строке, затем каждую часть по запятой
                lines = image_urls_str.splitlines()
                for line in lines:
                    urls_in_line = [url.strip() for url in line.split(',') if url.strip()]
                    image_urls_list.extend(urls_in_line)
            
            # Убедимся, что все элементы в списке - это действительные URL, или пустые строки,
            # и передаем этот список в predict_success
            # Здесь можно добавить более строгую валидацию URL, если нужно
            cleaned_data['project_image'] = image_urls_list # Теперь project_image будет списком URL

            # --- ВЫЗОВ AI-МОДЕЛИ ---  
            # Передаем очищенные данные формы напрямую в функцию предсказания.  
            # Функция predict_success теперь сама обрабатывает все преобразования.  
            ai_prediction_result, _ = predict_success(cleaned_data) # Получаем результат и DataFrame  
            messages.info(request, ai_prediction_result)  

            # --- СОХРАНЕНИЕ ПРОЕКТА В БАЗУ ДАННЫХ ---  
            # Логика сохранения данных в базу данных остается практически без изменений.  
            # Теперь image_db_value будет корректно сохранять все ссылки, разделенные запятыми
            image_db_value = ", ".join(image_urls_list) # Сохраняем список URL в виде строки через запятую
            
            project_instance = Project(  
                name=cleaned_data['name'],  
                blurb=cleaned_data['blurb'],  
                country=cleaned_data['country'],  
                usd_goal=cleaned_data['usd_goal'],  
                campaign_days=cleaned_data['campaign_days'],  
                prelaunch_activated=cleaned_data['prelaunch_activated'],  
                creation_date=cleaned_data['creation_date'],  
                creation_time=cleaned_data['creation_time'],  
                launch_date=cleaned_data['launch_date'],  
                launch_time=cleaned_data['launch_time'],  
                description=cleaned_data['description'],  
                images=image_db_value,  # Теперь это строка со всеми URL
                video=cleaned_data['video'] if cleaned_data['video'] else ""  
            )  

            project_instance.save()  
            messages.success(request, 'Данные проекта успешно сохранены!')  
            return redirect('ai_to_user:predict_form')  

        else:  
            messages.error(request, 'Пожалуйста, исправьте ошибки в форме.')  
    else:  
        form = ProjectPredictionForm()  

    context = {  
        'form': form,  
        'prediction_result': ai_prediction_result,  
    }  
    return render(request, 'ai_to_user/main.html', context)