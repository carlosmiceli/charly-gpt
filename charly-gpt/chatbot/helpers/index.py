import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import tiktoken
import hashlib
import re

def clean_up_text(text):
    # Remove unicode characters and special characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove extra whitespace and newline characters
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

def tiktoken_len(text):
    
    tokenizer = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(tokenizer.encode(text))
    
    # tokens = tokenizer.encode(
    #     text,
    #     disallowed_special=()
    # )
    return num_tokens

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

def process_and_upsert_chunks(text_chunks, metadata, url_id, embed_function, engine, index):
    metadatas = []
    ids = [f"{i + 1}-{url_id}" for i in range(len(text_chunks))]
    for i, chunk in enumerate(text_chunks):
        chunk_metadata = {
            "text": chunk,
            "chunk": i + 1,
            **metadata
        }

        metadatas.append(chunk_metadata)

    from tqdm.auto import tqdm
    from time import sleep

    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(text_chunks), batch_size)):
        # find end of batch
        i_end = min(len(text_chunks), i+batch_size)

        text_batch = text_chunks[i:i_end]
        meta_batch = metadatas[i:i_end]
        ids_batch = ids[i:i_end]

        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = embed_function(input=text_batch, engine=engine)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = embed_function(input=text_batch, engine=engine)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]

        vectors = list(zip(ids_batch, embeds, meta_batch))

        # upsert embeddings
        index.upsert(vectors=vectors)

def complete(prompt):
    # query text-davinci-003
    import openai
    # openai.api_key = os.environ.get('OPENAI_API_KEY')
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.5,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )

    return res['choices'][0]['text']

def create_full_prompt(query, query_response):

    # get relevant context
    contexts = [
        x['metadata']['text'] for x in query_response['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on your memory, and the instructions and relevant text excerpts below.\n\n"+
        "Instructions:\n\n"+
        "You are my coding tutor and mentor! You're here to assist me in my coding journey. If I provide a URL, you'll analyze the content of the page and provide me with a summary and useful information tailored to my personal goals.\n"+
        "As my coding companion, you won't apologize for not knowing something. If you're unsure about an answer, you'll let me know by saying 'I don't know' and explain the reasons behind it. Your ultimate goal is to ensure you provide me with accurate information.\n"+
        "You're capable of understanding both English and Spanish. I'm free to ask questions in either language—you'll respond accordingly. And when it's beneficial and appropriate, you'll leverage Google to gather additional information in order to present me with the best possible answers.\n"+
        "Additionally, you possess the ability to remember our past conversations, which enables you to take into account our previous interactions when offering answers. So, I can go ahead and ask you anything related to coding or programming!\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )

    limit = 3750
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

# Define the function to retrieve content from the vector database using semantic search
def retrieve_content_from_database(query):

    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY', ''), environment='us-west4-gcp-free')
    
    embed = OpenAIEmbeddings()
    index = os.environ.get('PINECONE_INDEX_NAME', '')
    vectorstore = Pinecone(index, embed.e, text_field="text")

    # Use semantic search to retrieve relevant content from the vector database
    response = vectorstore.search(query, search_type=SearchType.SIMILARITY)

    # Extract and return the retrieved content
    retrieved_content = []
    for result in response['results']:
        retrieved_content.append(result['values']['text'])

    return retrieved_content

# Define the function to retrieve content from SERP using the SERP API
def retrieve_content_from_serp(query):
    return
    # Use the SERP API to retrieve relevant content
    # Return the retrieved content

# Define the function to generate a response using retrieved content from both sources
def generate_response_with_serp_and_database(query, use_database, use_serp):
    response = ""

    if use_database:
        database_content = retrieve_content_from_database(query)
        response += "Content from database: " + database_content

    if use_serp:
        serp_content = retrieve_content_from_serp(query)
        response += "Content from SERP: " + serp_content

    return response

# Define the function to analyze content from a URL
def analyze_url(url, query, save_to_database, use_serp):
    try:
        url_id = str(hash_url_to_id(url))

        index = pinecone.Index(os.environ.get('PINECONE_INDEX_NAME', ''))

        # ... (other code for loading and processing the document from the URL)

        if save_to_database:
            # Store content in the vector database
            process_chunks(chunks, embed, index, batch_limit)

        # Generate response using retrieved content
        response = generate_response_with_serp_and_database(query, save_to_database, use_serp)

        return response

    except Exception as e:
        # Handle exceptions
        return {'error': 'Error occurred while analyzing the URL.'}

# Define the function to analyze content from a file
def analyze_file(file_content, query, save_to_database, use_serp):
    try:
        # ... (other code for loading and processing the file content)

        if save_to_database:
            # Store content in the vector database
            process_chunks(chunks, embed, index, batch_limit)

        # Generate response using retrieved content
        response = generate_response_with_serp_and_database(query, save_to_database, use_serp)

        return response

    except Exception as e:
        # Handle exceptions
        return {'error': 'Error occurred while analyzing the file.'}

