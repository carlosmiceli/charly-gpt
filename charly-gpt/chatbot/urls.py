# document_analyzer_app/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("", views.form, name="form"),
    path("delete_url_namespace", views.delete_url_namespace, name="delete_url_namespace"),
    path("analyze_document/", views.analyze_document, name="analyze_document"),
    path('chat_agent/', views.chat_agent, name='chat_agent'),
]
