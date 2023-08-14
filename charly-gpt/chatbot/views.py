import os
import logging
import json
import traceback
import pinecone 
from django.shortcuts import render
from django.http import JsonResponse
from langchain import PromptTemplate
from langchain.document_loaders import BrowserlessLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
# from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
import tiktoken
import hashlib
from tqdm.auto import tqdm
from uuid import uuid4

tiktoken.encoding_for_model('gpt-3.5-turbo')
tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# Get an instance of a logger
logger = logging.getLogger(__name__)

# Initialize OpenAI
openai_api_key = os.environ.get('OPENAI_API_KEY', '')
llm = OpenAI(openai_api_key=openai_api_key)
max_token_limit=4097

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=openai_api_key,
)

# Initialize Pinecone
pinecone.init(api_key=os.environ.get('PINECONE_API_KEY', ''), environment='us-west4-gcp-free')


# Create an instance of the SerpAPIWrapper if the API key is available
serp_api_key = os.environ.get('SERP_API_KEY', '')

serpapi = SerpAPIWrapper(serpapi_api_key=serp_api_key)

# # Initialize the chat model
# chat_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1)

# Initialize the conversation chain
conversation_summary_buffer_window = ConversationChain(
    llm=llm, 
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=650
))

# Initialize the prompt template
prompt_template = """
I'm your coding tutor and mentor! I'm here to assist you in your coding journey. 
If you provide a URL, I'll analyze the content of the page and provide you with a summary and useful information for your personal goals based on it. 
As your coding companion, I won't apologize for not knowing something. 
If I'm unsure about an answer, I'll let you know by saying "I don't know" and explain why. My goal is to provide accurate information. 
I'm capable of understanding both English and Spanish. Feel free to ask questions in either language, and I'll respond accordingly. 
When it's appropriate and helpful, I'll use Google to gather additional information to provide you with the best possible answers. 
I also have the ability to remember our past conversations. This allows me to consider our previous interactions when providing answers. 
Now, go ahead and ask me anything related to coding or programming!

Context: {document}

Question: {query}
AI Response: """

#ADD EXAMPLES TO THE PROMPT

def hash_url_to_id(url):
                # Create a hashlib object using SHA-256
                hasher = hashlib.sha256()

                # Update the hasher with the URL's bytes
                hasher.update(url.encode('utf-8'))

                # Get the hexadecimal digest
                hash_digest = hasher.hexdigest()[:16]

                # Convert the hexadecimal digest to an integer
                url_id = int(hash_digest, 16)

                return url_id

def form(request):
    return render(request, 'form.html')

def delete_url_namespace(request):
    if request.method == 'POST':
        index = pinecone.Index(os.environ.get('PINECONE_INDEX_NAME', ''))
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)
        try:
            logger.info(index)
            url = data.get('url', '')
            namespace_id = str(hash_url_to_id(url))
            delete_namespace = index.delete(
                namespace=namespace_id,
                delete_all=True
            )
            logger.info(f"Query response: {delete_namespace}")
            return JsonResponse({'success': 'Namespace deleted successfully.'}, status=200)
        except Exception as e:
            logger.error(f"Error while querying index: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({'error': 'Error occurred while querying index.'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method.'})

# Analyze Document view
def analyze_document(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

        try:
            # Load data from the URL
            query = data.get('input', '')
            url = data.get('url', '')
            
            namespace_id = str(hash_url_to_id(url))

            index = pinecone.Index(os.environ.get('PINECONE_INDEX_NAME', ''))
            logger.info(index.describe_index_stats())

            metadata = {
                'namespace_id': namespace_id,
                'url': url,
            }

            # Initialize the BrowserlessLoader
            browserless_loader = BrowserlessLoader(
                api_token=os.environ.get('BROWSERLESS_API_KEY', ''),
                urls=[url],  # Load a single URL
            )
            documents = browserless_loader.load()

            # # You can now process the loaded Documents as needed
            page_content = documents[0].page_content

            # prompt = PromptTemplate(
            #     input_variables=['document', 'query'],
            #     template=prompt_template
            # )

            # # logger.info(prompt)

            # # logger.info(llm(prompt.format(document=page_content, query=input_text)))

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,
                chunk_overlap=20,
                length_function=tiktoken_len,
                separators=["\n\n", "\n", " ", ""]
            )

            batch_limit=100

            chunks = text_splitter.split_text(page_content)

            def process_and_upsert_chunks(text_chunks, embed, index, batch_limit):

                metadatas = []
                for i, chunk in enumerate(text_chunks):
                    chunk_metadata = {
                        "chunk": i, "text": chunk, **metadata
                    }
                    metadatas.append(chunk_metadata)

                remaining_chunks = len(text_chunks)

                while remaining_chunks > 0:
                    current_batch_size = min(remaining_chunks, batch_limit)

                    current_text_batch = text_chunks[:current_batch_size]
                    current_metadata_batch = metadatas[:current_batch_size]
                    current_ids = [f"{i + 1}-{namespace_id}" for i in range(current_batch_size)]

                    current_embeds = embed.embed_documents(current_text_batch)
                    vectors=[
                        {
                            "id": current_ids[i],
                            "values": current_embeds[i],
                            "metadata": current_metadata_batch[i]
                        }
                        for i in range(len(current_ids))
                    ]
                    index.upsert(vectors=vectors, namespace=namespace_id)

                    text_chunks = text_chunks[current_batch_size:]
                    metadatas = metadatas[current_batch_size:]

                    remaining_chunks -= current_batch_size

                logger.info(f"IDs: {current_ids}")

            text_field = 'text'
            # Call the function with your data and parameters
            process_and_upsert_chunks(chunks, embed, index, batch_limit)

            vectorstore = Pinecone(index, embed.embed_query, text_field)
            logger.info(f"Vectorstore: {vectorstore}")

            results = vectorstore.similarity_search(query, k=3)  # return 3 most relevant docs
            logger.info(f"Results: {results}")

            





            return JsonResponse(results, safe=False)

        except Exception as e:
            # Handle any potential exceptions related to document loading and processing
            logger.error(f"Error while analyzing the document: {str(e)}")
            logger.error(traceback.format_exc())  # Log the traceback
            return JsonResponse({'error': 'Error occurred while analyzing the document.'}, status=500)

    return JsonResponse({'error': 'Invalid request method.'})

# Chat Agent view
def chat_agent(request):
    return 1
#     if request.method == 'POST':
#         try:
#             data = json.loads(request.body)
#         except json.JSONDecodeError:
#             return JsonResponse({'error': 'Invalid JSON data.'}, status=400)

#         input_text = data.get('input', '')
#         use_serpapi = data.get('use_serpapi', False)  # Retrieve from payload

#         # Create a memory for the conversation
#         memory = ConversationBufferMemory(
#             memory_key="chat_history", return_messages=True)

#         # Initialize the agent
#         tools = []

#         if use_serpapi:

#             def perform_serpapi_search(input_text):
#                 try:
#                     search_results = serpapi.results(input_text)
#                 except Exception as e:
#                     logger.error(
#                         f"Error while fetching search results from SerpAPI: {str(e)}")
#                     return []

#                 formatted_results = []

#                 if "organic_results" in search_results:
#                     organic_results = search_results["organic_results"]

#                     for result in organic_results:
#                         title = result.get("title", "No Title")
#                         link = result.get("link", "#")
#                         snippet = result.get("snippet", "No Snippet")

#                         formatted_result = {
#                             "title": title,
#                             "link": link,
#                             "snippet": snippet
#                         }

#                         formatted_results.append(formatted_result)

#                 return formatted_results

#             logger.info("Using SerpAPI for current search tool.")

#             # Create the SerpAPI tool
#             serpapi_tool = Tool(name="SerpAPI Search Results",
#                                 func=perform_serpapi_search, description="SerpAPI search results")

#             # Append the SerpAPI tool to the tools list
#             tools.append(serpapi_tool)

#         # Initialize the agent with the tools
#         agent_chain = initialize_agent(
#             tools, chat_model, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

#         logger.info(f"Agent chain: {agent_chain}")

#         # Run the agent with user input
#         result = agent_chain.run(input=input_text)

#         logger.info(f"Result from chat agent: {result}")

#         # If the result is a string, try to parse it as JSON
#         if isinstance(result, str):
#             try:
#                 result = json.loads(result)
#             except json.JSONDecodeError:
#                 # If the result is not valid JSON, treat it as a successful response
#                 final_answer = result
#             else:
#                 # If the result is valid JSON, extract the final answer
#                 final_answer = result.get(
#                     "action_input", "I'm sorry, there was an issue with the agent's response.")
#         else:
#             # If the result is already a dictionary, extract the final answer
#             final_answer = result.get(
#                 "action_input", "I'm sorry, there was an issue with the agent's response.")

#         # Return the answer as a JSON response
#         return JsonResponse({'answer': final_answer})

#     return JsonResponse({'error': 'Invalid request method.'})
