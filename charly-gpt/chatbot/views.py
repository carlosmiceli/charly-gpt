from django.shortcuts import render

import os
from django.http import HttpResponse
from langchain.llms import OpenAI
from .modules.document_analyzer import DocumentAnalyzer
import logging

llm = OpenAI(openai_api_key=os.environ['OPENAI_API_KEY'])

logger = logging.getLogger(__name__)  # Use __name__ to get the logger for the current module

def index(request):
    logger.info("llm: " + str(llm))
    return render(request, 'form.html')

def analyze_document(request):
    doc_url = 'https://paper.dropbox.com/doc/My-Journey-to-Coding-Mastery--B9KApGZY3U8xGROPGinEHMmhAg-kDdrF9BRDEx3KXjH6V8tp'
    analyzer = DocumentAnalyzer(doc_url)
    data = analyzer.get_analysis_results()
    # Process and format the analysis results as needed
    data