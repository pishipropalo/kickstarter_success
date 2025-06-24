# ai_to_user/urls.py
from django.urls import path
from . import views

app_name = 'ai_to_user'

urlpatterns = [
    # This line will now map the empty path ('') within the 'ai_to_user' namespace
    # to your predict_view. Since your main urls.py includes this app at '',
    # this means http://127.0.0.1:8000/ will correctly go to predict_view.
    path('', views.predict_view, name='predict_form'), # <-- CHANGE THIS LINE
    # If you also want /predict/ to work, you can add:
    # path('predict/', views.predict_view, name='predict_form_alias'), # Or just use the '' path
]