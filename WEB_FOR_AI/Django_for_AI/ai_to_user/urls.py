# ai_to_user/urls.py
from django.urls import path
from . import views

app_name = 'ai_to_user'

urlpatterns = [
    path('', views.predict_view, name='predict_form'), 
    
]
