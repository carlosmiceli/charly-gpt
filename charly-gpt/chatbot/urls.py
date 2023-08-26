# document_analyzer_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("", views.form, name="form"),
    path("delete_url", views.delete_url, name="delete_url"),
    path('chat_agent', views.chat_agent, name='chat_agent'),
]
