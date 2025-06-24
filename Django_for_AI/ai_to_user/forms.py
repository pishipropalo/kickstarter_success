from django import forms

class ProjectPredictionForm(forms.Form):
    name = forms.CharField(
        label="Название проекта",
        max_length=200,
        widget=forms.TextInput(attrs={'placeholder': 'Например, Киберпанк Одиссея'}),
        help_text="Полное название вашего игрового проекта."
    )
    blurb = forms.CharField(
        label="Краткое описание проекта",
        widget=forms.Textarea(attrs={'rows': 2, 'placeholder': 'Например, Ретро-пиксельная RPG, где вы играете за разумный торговый автомат.'}),
        help_text="Краткое, цепляющее описание вашей игры."
    )
    country = forms.CharField(
        label="Страна происхождения",
        max_length=50,
        widget=forms.TextInput(attrs={'placeholder': 'Например, The United Kingdom, Poland, The United States'}),
        help_text="Страна, из которой запускается ваш проект."
    )
    usd_goal = forms.FloatField(
        label="Цель финансирования (USD)",
        min_value=0.01,
        widget=forms.NumberInput(attrs={'step': '0.01', 'placeholder': 'Например, 25000.00'}),
        help_text="Общая сумма в долларах США, которую вы планируете собрать."
    )
    campaign_days = forms.IntegerField(
        label="Длительность кампании (дней)",
        min_value=1,
        max_value=100000000, # Большинство платформ ограничивают до 60 дней, но можно изменить по желанию
        widget=forms.NumberInput(attrs={'placeholder': 'Например, 30, 45'}),
        help_text="Сколько дней будет длиться ваша краудфандинговая кампания."
    )
    prelaunch_activated = forms.BooleanField(
        label="Предстартовая кампания активирована?",
        required=False, # Чекбокс необязателен
        help_text="Отметьте, если планируете проводить предстартовую кампанию."
    )
    creation_date = forms.DateField(
        label="Дата создания проекта",
        widget=forms.DateInput(attrs={'type': 'date'}),
        help_text="Дата первоначального создания вашего проекта."
    )
    creation_time = forms.TimeField(
        label="Время создания проекта",
        widget=forms.TimeInput(format='%H:%M', attrs={'type': 'time', 'step': '60'}),
        help_text="Время первоначального создания вашего проекта (часы:минуты)."
    )
    launch_date = forms.DateField(
        label="Дата запуска кампании",
        widget=forms.DateInput(attrs={'type': 'date'}),
        help_text="Планируемая дата запуска вашей краудфандинговой кампании."
    )
    launch_time = forms.TimeField(
        label="Время запуска кампании",
        widget=forms.TimeInput(format='%H:%M', attrs={'type': 'time', 'step': '60'}),
        help_text="Планируемое время запуска вашей краудфандинговой кампании (часы:минуты)."
    )
    description = forms.CharField(
        label="Полное описание проекта",
        widget=forms.Textarea(attrs={'rows': 15, 'placeholder': 'Подробно опишите свою игру: жанр, механики, сюжет, художественный стиль, целевую аудиторию и уникальные особенности.'}),
        help_text="Детальное текстовое описание вашего проекта."
    )
    project_image = forms.CharField(   
        label="Ссылки на изображения проекта",  
        required=False,  
        widget=forms.Textarea(attrs={'rows': 4, 'placeholder': 'https://example.com/image1.jpg, https://example.com/image2.png'}),  
        help_text="Введите прямые ссылки на изображения вашего проекта, разделяя их запятыми или с новой строки."  
    ) 
    video = forms.URLField(
        label="Ссылки на видео",
        required=False,
        widget=forms.URLInput(attrs={'placeholder': 'Например, https://youtube.com/watch?v=your_video_id'}),
        help_text="Ссылка на трейлер или геймплейное видео вашего проекта на YouTube и т.д."
    )