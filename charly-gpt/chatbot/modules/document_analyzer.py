from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

class DocumentAnalyzer:
    def __init__(self, urls):
        self.urls = urls

    def load_documents(self):
        loaders = []
        for url in self.urls:
            loader = WebBaseLoader(url)
            data = loader.load()
            loaders.append(data)
        return loaders

    def analyze_document(self, query):
        data = self.load_documents()
        index = VectorstoreIndexCreator().from_loaders(data)
        result = index.query(query)
        return result

    def get_analysis_results(self, query):
        result = self.analyze_document(query)
        return result
