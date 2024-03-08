# Description: This file contains the main code for the RAG application
# Im@Author: Sujan Chandra Roy
# Date: 2024-03-08
#Version: 1.0
# =============================================================================
# Importing the required libraries
# =============================================================================
import torch
from upstash_vector import Vector, Index
from datasets import load_dataset
from tqdm import tqdm, trange
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

notebook_login()



# =============================================================================
# Loading the dataset
# =============================================================================
data=load_dataset("HuggingFaceTB/cosmopedia","stanford",split="train")
# =============================================================================
# Convert the dataset into pandas dataframe and store it in a csv file
# =============================================================================
data = data.to_pandas()
data.to_csv("standford_dataset.csv")
data.head()
print(data.head())

loader = CSVLoader("/home/vv070/Desktop/LangchainApp/RAG_Application/standford_dataset.csv")
data = loader.load()

# =============================================================================
# Spilt the documnents inside the data into smaller chunks that can fit into our models context window
# =============================================================================
text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_spliter.split_documents(data)
# =============================================================================

# =============================================================================
# Generating Embeddings with Sentence Transformers Model    

modelpath = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
    model_name = modelpath, 
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs)

# Generate the embeddings for each chunk of text
chunk_embeddings = []

for doc in docs:
    chunk_embedding = embeddings.encode(doc)
    chunk_embeddings.append(chunk_embedding)
# =============================================================================
    
# Store the embeddings in a upstash vector
UPSTASH_Vector_Rest_URL = "Your own URL of upstash vector"
UPSTASH_Vector_Rest_Token = "Your own Token of upstash vector"

vectors = []

# Generate the vectors in batches of 10
batch_count = 10

for i in trange(0, len(chunk_embeddings), batch_count):
    batch  = chunk_embeddings[i:i+batch_count]

    embeddings = chunk_embedding[batch]

    for i, chunk in enumerate(batch):
        vec = Vector(id = f"chunk-{i}", vector = embeddings[i], metadata={"text":chunk})

        vectors.append(vec)

index = Index(url = UPSTASH_Vector_Rest_URL, token = UPSTASH_Vector_Rest_Token)

index.upsert(vectors)

# =============================================================================
# Initalize the tokenizer and model

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b",padding=True, truncation=True,max_length=512)

# =============================================================================
# Create a text generation pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_tensors= "pt",
    max_length=512,
    max_new_tokens=512,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)

# =============================================================================
# Initalize the LLM with pipeline and model awards

llm = HuggingFacePipeline(pipeline= pipe, 
                          model_kwargs = {"temperature": 0.7, "top_k": 50, "top_p": 0.95, "max_length": 512}
                          )

# =============================================================================
# Query the LLM with a question and get the answer

def ask_question(question):
    # Get the embeddings for the question
    question_embedding = embeddings.encode(doc)
    
    # Search for similar vectors
    res = index.query(vector= question_embedding, top_k=5, return_metadata=True)
    # collect the results in a context
    context = "\n".join([r.metadata["text"] for r in res])

    prompt = f"Question: {question}\nContext: {context}"
    
    # Generate the answer using llm
    answer = llm(prompt)

    # Return the answer
    return answer[0]["generated_text"]

ask_question("Write an educational story for younng children")
print(ask_question)

