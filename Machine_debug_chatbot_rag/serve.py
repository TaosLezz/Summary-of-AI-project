from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import chromadb
from flask_cors import CORS
from rag.core import RAG
from embeddings import OpenAIEmbedding
from semantic_router import SemanticRouter, Route
from semantic_router.samples import productsSample, chitchatSample
import openai
from reflection import Reflection
import ollama
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(
    filename="app.log",         # Tên file log
    level=logging.INFO,         # Ghi log từ mức INFO trở lên
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load environment variables from .env file
load_dotenv()
# Access the key
LLM_KEY = os.getenv('OLLAMA_KEY')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL') or 'keepitreal/vietnamese-sbert'
# OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
# OPEN_AI_EMBEDDING_MODEL = os.getenv('OPEN_AI_EMBEDDING_MODEL') or 'text-embedding-3-small'

# OpenAIEmbedding(OPEN_AI_KEY)





# --- Semantic Router Setup --- #

PRODUCT_ROUTE_NAME = 'products'
CHITCHAT_ROUTE_NAME = 'chitchat'

# openAIEmbeding = OpenAIEmbedding(apiKey=OPEN_AI_KEY, dimensions=1024, name=OPEN_AI_EMBEDDING_MODEL)
# ollamaEmbedding = ollama.embeddings(model="nomic-embed-text")
openAIEmbeding = SentenceTransformer(EMBEDDING_MODEL)
productRoute = Route(name=PRODUCT_ROUTE_NAME, samples=productsSample)
chitchatRoute = Route(name=CHITCHAT_ROUTE_NAME, samples=chitchatSample)
semanticRouter = SemanticRouter(openAIEmbeding, routes=[productRoute, chitchatRoute])

# --- End Semantic Router Setup --- #

# Initialize ChromaDB client
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="documents")

# --- Set up LLMs --- #

llm = openai.OpenAI(
    base_url="http://192.168.1.67:11434/v1",
    api_key = LLM_KEY
)

# --- End Set up LLMs --- #

# --- Relection Setup --- #

gpt = openai.OpenAI(
    base_url="http://192.168.1.67:11434/v1",
    api_key = LLM_KEY
)
reflection = Reflection(llm=gpt)

# --- End Reflection Setup --- #

app = Flask(__name__)
CORS(app)


# Initialize RAG
rag = RAG(
    llm=llm,
    chromaPath="./chroma_db",
    embeddingName='sentence-transformers/all-MiniLM-L6-v2',
)

def process_query(query):
    return query.lower()

@app.route('/api/search', methods=['POST'])
def handle_query():
    data = list(request.get_json())

    query = data[-1]["parts"][0]["text"]

    query = process_query(query)

    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # get last message
    
    guidedRoute = semanticRouter.guide(query)[1]
    user_input = data[0]["parts"][0]["text"]
    print("data", data[0]["parts"][0]["text"])
    if guidedRoute == PRODUCT_ROUTE_NAME:
        # Decide to get new info or use previous info
        # Guide to RAG system
        print("Guide to RAGs")
        print("data", data)
        reflected_query = reflection(data)

        # print('====query', query)
        print('reflected_query', reflected_query)

        query = reflected_query
        print("input user_input", user_input)
        source_information = rag.enhance_prompt(query).replace('<br>', '\n')
        # source_information = rag.vector_search(query, 4).replace('<br>', '\n')
        print('source_information', source_information)
        combined_information = f"Become an expert in troubleshooting machine failures in industrial factories. Engineer's question: {user_input}\nAnswer the question in the same language as the engineer's question, providing a technical response based on the following defective product information: {source_information}."
        prompt =""""You are an expert in troubleshooting machine failures in industrial factories. Engineer's question: {user_input}
If the question contains specific details related to a known defective product, provide a precise technical response using only the relevant information from the following source: {source_information}.
If the question cannot be answered based on the provided defective product information, respond with 'I don't know'."*"""
        messages = [
            {"role": "system", "content": "You are an expert in troubleshooting machine failures."},
            {"role": "user", "content": combined_information}
        ]
        response = rag.generate_content(messages)
        logging.info(f"Generated response: {response}")
        print(f"Generated response: {response}")
    else:
        # Guide to LLMs
        print("Guide to LLMs")
        response = llm.chat.completions.create(
            model="qwen2.5:32b-instruct-q4_0",
            messages=[
                {"role": "user", "content": user_input},
            ]
        ).choices[0].message.content
    # print('====data', data)
    
    return jsonify({
        'parts': [
            {
            'text': response,
            }
        ],
        'role': 'model'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
