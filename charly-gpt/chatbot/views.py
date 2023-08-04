import os
import logging
import requests
import json
import traceback
from django.shortcuts import render
from django.http import JsonResponse
from langchain.document_loaders import BrowserlessLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.utilities import SerpAPIWrapper
from langchain.llms import OpenAI

# Initialize OpenAI and the logger
openai_api_key = os.environ.get('OPENAI_API_KEY', '')
llm = OpenAI(openai_api_key=openai_api_key)
logger = logging.getLogger(__name__)

# Create an instance of the SerpAPIWrapper if the API key is available
serp_api_key = os.environ.get('SERP_API_KEY', '')
search = SerpAPIWrapper(serpapi_api_key=serp_api_key) if serp_api_key else None

# Initialize the chat model
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def index(request):
    return render(request, 'form.html')

# Analyze Document view
# Analyze Document view
def analyze_document(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            input_data = data.get('input', '')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

        try:
            # Load data from the URL
            logger.info(f"Loading data from URL: {input_data}")
            response = requests.post(
                "https://chrome.browserless.io/scrape",
                params={
                    "token": os.environ.get('BROWSERLESS_API_KEY', ''),
                },
                json={
                    "url": input_data,
                    "elements": [{"selector": "body"}],
                },
            )
            # Log the API response data
            response_data = response.json()
            logger.info(f"API Response: {response_data}")

            # Check if the response is a list and extract the page content accordingly
            if isinstance(response_data, list):
                if response_data:
                    if "results" in response_data[0] and response_data[0]["results"]:
                        page_content = response_data[0]["results"][0].get("text", "")
                    else:
                        return JsonResponse({'error': 'No "results" key found in the API response.'}, status=500)
                else:
                    return JsonResponse({'error': 'Empty response from the Browserless API.'}, status=500)
            else:
                return JsonResponse({'error': 'Invalid response format from the Browserless API.'}, status=500)

            # Split the Document into chunks for embedding and vector storage
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            all_splits = text_splitter.split_text(page_content)

            # Store the splits in the vectorstore
            from langchain.vectorstores import Chroma
            vectorstore = Chroma.from_texts(texts=all_splits, embedding=OpenAIEmbeddings())

            # Retrieve relevant splits for the given question
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(chat_model, retriever=retriever)
            result = qa_chain({"query": input_data})

            # Get the answer from the result
            if "result" in result and result["result"]:
                answer = result["result"][0]["text"]
                logger.info(f"Answer from document analysis: {answer}")
            else:
                answer = "No answer found"

            # Return the answer as a JSON response
            return JsonResponse({'answer': answer})

        except Exception as e:
            # Handle any potential exceptions related to document loading and processing
            logger.error(f"Error while analyzing the document: {str(e)}")
            logger.error(traceback.format_exc())  # Log the traceback
            return JsonResponse({'error': 'Error occurred while analyzing the document.'}, status=500)

    return JsonResponse({'error': 'Invalid request method.'})

# Chat Agent view
def chat_agent(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('input', '')
            use_serpapi = data.get('use_serpapi', False)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

        # Create a memory for the conversation
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize the agent
        tools = []
        if use_serpapi and search:
            logger.info("Using SerpAPI for current search tool.")
            tools.append(
                Tool(
                    name="Current Search",
                    func=search.run,
                    description="useful for when you need to answer questions about current events or the current state of the world"
                )
            )
        # Add more tools here if needed

        agent_chain = initialize_agent(tools, chat_model, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

        # Run the agent with user input
        result = agent_chain.run(input=user_input)

        logger.info(f"Result from chat agent: {result}")

        # If the result is a string, try to parse it as JSON
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return JsonResponse({'error': 'Invalid JSON data in agent response.'}, status=500)

        # Get the final answer from the result
        final_answer = result.get("action_input", "I'm sorry, there was an issue with the agent's response.")

        # Return the answer as a JSON response
        return JsonResponse({'answer': final_answer})

    return JsonResponse({'error': 'Invalid request method.'})

