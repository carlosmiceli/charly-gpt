import os
import logging
import json
import traceback
from django.shortcuts import render
from django.http import JsonResponse
from langchain.document_loaders import BrowserlessLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from serpapi import GoogleSearch
from langchain.utilities import SerpAPIWrapper


# Initialize OpenAI and the logger
openai_api_key = os.environ.get('OPENAI_API_KEY', '')
llm = OpenAI(openai_api_key=openai_api_key)
logger = logging.getLogger(__name__)

# Create an instance of the SerpAPIWrapper if the API key is available
serp_api_key = os.environ.get('SERP_API_KEY', '')

serpapi = SerpAPIWrapper(serpapi_api_key=serp_api_key)

# Initialize the chat model
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def index(request):
    return render(request, 'form.html')

# Analyze Document view
def analyze_document(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

        try:
            # Load data from the URL
            input_text = data.get('input', '')
            url = data.get('url', '')

            # Initialize the BrowserlessLoader
            browserless_loader = BrowserlessLoader(
                api_token=os.environ.get('BROWSERLESS_API_KEY', ''),
                urls=[url],  # Load a single URL
                text_content=True  # Set to False if you want to load raw HTML content
            )

            # Load Documents from the URL
            documents = browserless_loader.load()

            # You can now process the loaded Documents as needed
            page_content = documents[0].page_content

            # Split the Document into chunks for embedding and vector storage
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            all_splits = text_splitter.split_text(page_content)

            # Store the splits in the vectorstore
            vectorstore = Chroma.from_texts(texts=all_splits, embedding=OpenAIEmbeddings())

            # Define the query
            query = input_text

            # Retrieve relevant documents for the query using the retriever (vectorstore)
            retriever = vectorstore.as_retriever()

            # Initialize the RetrievalQA with the retriever
            qa_chain = RetrievalQA.from_chain_type(chat_model, retriever=retriever)

            # Use the retrieved text as context for the query
            result = qa_chain({"query": query, "context": page_content})

            # Get the answer from the result
            if "result" in result and result["result"]:
                answer = result["result"]
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
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

        input_text = data.get('input', '')
        use_serpapi = data.get('use_serpapi', False)  # Retrieve from payload

        # Create a memory for the conversation
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize the agent
        tools = []

        if use_serpapi:
            
            def perform_serpapi_search(input_text):
                try:
                    search_results = serpapi.results(input_text)
                except Exception as e:
                    logger.error(f"Error while fetching search results from SerpAPI: {str(e)}")
                    return []

                formatted_results = []

                if "organic_results" in search_results:
                    organic_results = search_results["organic_results"]

                    for result in organic_results:
                        title = result.get("title", "No Title")
                        link = result.get("link", "#")
                        snippet = result.get("snippet", "No Snippet")

                        formatted_result = {
                            "title": title,
                            "link": link,
                            "snippet": snippet
                        }

                        formatted_results.append(formatted_result)

                return formatted_results

            logger.info("Using SerpAPI for current search tool.")
            
            # Create the SerpAPI tool
            serpapi_tool = Tool(name="SerpAPI Search Results", func=perform_serpapi_search, description="SerpAPI search results")
            
            # Append the SerpAPI tool to the tools list
            tools.append(serpapi_tool)

        # Initialize the agent with the tools
        agent_chain = initialize_agent(tools, chat_model, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

        logger.info(f"Agent chain: {agent_chain}")

        # Run the agent with user input
        result = agent_chain.run(input=input_text)

        logger.info(f"Result from chat agent: {result}")

        # If the result is a string, try to parse it as JSON
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                # If the result is not valid JSON, treat it as a successful response
                final_answer = result
            else:
                # If the result is valid JSON, extract the final answer
                final_answer = result.get("action_input", "I'm sorry, there was an issue with the agent's response.")
        else:
            # If the result is already a dictionary, extract the final answer
            final_answer = result.get("action_input", "I'm sorry, there was an issue with the agent's response.")

        # Return the answer as a JSON response
        return JsonResponse({'answer': final_answer})

    return JsonResponse({'error': 'Invalid request method.'})

