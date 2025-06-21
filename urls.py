from django.urls import path
from . import views

urlpatterns = [
    path("", views.index),
    path("signup", views.signup, name="signup"),
    path("login", views.login, name="login"),
    path("logout", views.logout, name="logout"),
    path("predict_with_input_form", views.predict_with_input_form, name="predict_with_input_form"),
    path("use_app", views.use_app, name="use_app"),
    path('predict_with_file', views.predict_with_file, name='predict_with_file')  
]