from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFacePipeline  # Updated import
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
from transformers import pipeline       #type:ignore
 
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY          #type:ignore

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


# Initialize Hugging Face pipeline for text generation (e.g., GPT-2)
generator = pipeline("text-generation", model="gpt2", max_length=500, temperature=0.4, truncation=True, pad_token_id=50256)

# Wrap the Hugging Face pipeline in LangChain
llm = HuggingFacePipeline(pipeline=generator)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # Define your system prompt here
        ("human", "{input}"),  # This is the input from the user
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
