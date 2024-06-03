import os
import boto3

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import BedrockEmbeddings, OllamaEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_community.chat_models import BedrockChat, ChatOllama

def load_llm(use_llm: str):
  if use_llm == "openai":
    return ChatOpenAI(temperature=0, model_name=os.getenv("OPENAI_LLM"), streaming=True, api_key=os.getenv("OPENAI_API_KEY"))
  elif use_llm == "aws":
    session= boto3.Session(
      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    bedrock=session.client(
      region_name=os.getenv("AWS_DEFAULT_REGION")
    )
    return BedrockChat(
      client=bedrock,
      model_id=os.getenv("AWS_LLM"),
      model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
      streaming=True
    )

  elif use_llm == "ollama":
    return ChatOllama(
      temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_LLM"),
            streaming=True,
            top_k=10,
            top_p=0.3,  
            num_ctx=3072
    )
  return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True, api_key=os.getenv("OPENAI_API_KEY"))

def load_embedding_model (use_embedding_model: str, dimension=None) :
  if use_embedding_model == "ollama":
    embeddings = OllamaEmbeddings(
      base_url = os.getenv("OLLAMA_BASE_URL"),
      model = os.getenv("OLLAMA_LLM")
    )
    dimension = dimension if dimension != None else 4096
  elif use_embedding_model == "openai":
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
    dimension = dimension if dimension != None else 1536
  elif use_embedding_model == "aws":
    embeddings = BedrockEmbeddings()
    dimension = dimension if dimension != None else 1536
  return embeddings, dimension