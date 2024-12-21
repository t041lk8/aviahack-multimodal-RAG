from langchain.schema import HumanMessage, SystemMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
import logging
from tqdm import tqdm
import re
from typing import Callable, Union, Optional
from transliterate import translit
from transformers import AutoConfig, AutoTokenizer
from langchain.docstore.document import Document
import math
from joblib import Parallel, delayed, cpu_count
from typing import List
import torch

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFacePipeline

from accelerate import dispatch_model, infer_auto_device_map

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()


class Models:
    def __init__(
        self,
        embedder_model_name,
        reranker_model_name,
    ):
        self._embedder_model_name = embedder_model_name
        self._reranker_model_name = reranker_model_name

        self._embedder_config = AutoConfig.from_pretrained(embedder_model_name)
        self._embedder_tokenizer = AutoTokenizer.from_pretrained(embedder_model_name)

        self._reranker_config = AutoConfig.from_pretrained(reranker_model_name)
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)

    def length_function(self, text):
        max_tokens = max(
            len(
                self._embedder_tokenizer.encode(
                    text,
                    truncation=False,
                    padding=False,
                    max_length=self._embedder_config.max_position_embeddings,
                )
            ),
            len(
                self._reranker_tokenizer.encode(
                    text,
                    truncation=False,
                    padding=False,
                    max_length=self._reranker_config.max_position_embeddings,
                )
            ),
        )
        return max_tokens

EMBEDDER_MODEL_NAME = "BAAI/bge-m3"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL_NAME)

def split_markdown_by_headers(content):

    # Удаляем фразу <!-- image -->
    content = content.replace('<!-- image -->', '')

    # Разбиваем контент по заголовкам первого уровня
    segments = content.split('\n# ')
    segments = ['# ' + segment if idx != 0 else segment for idx, segment in enumerate(segments)]
    
    return segments

models = Models(
    "BAAI/bge-m3",
    "BAAI/bge-m3",
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=50, length_function=models.length_function
)
    
def process_batch(batch):
    return text_splitter.split_documents(batch)

def format_document(doc):
    return Document(
        page_content=doc
    )

def batch_docs(docs, n_jobs):
    batch_size = math.ceil(len(docs) / n_jobs)
    return [docs[i : i + batch_size] for i in range(0, len(docs), batch_size)]

def get_processed_docs():
    with open("ragdata.md", "r") as f:
        content = f.read()
        
    documents = split_markdown_by_headers(content)

    n_jobs = -1
    formatted_docs = Parallel(n_jobs=n_jobs)(
    delayed(format_document)(doc) for doc in tqdm(documents)
    )
    logger.info("Документы форматированы")

    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    batched_docs = batch_docs(formatted_docs, n_jobs)
    logger.info("Документы сгруппированы")

    docs = Parallel(n_jobs=n_jobs)(
        delayed(process_batch)(batch) for batch in tqdm(batched_docs)
    )
    logger.info("Документы обработаны")

    docs = [doc for document in docs for doc in document]
    return docs


def load_vectorestore_from_file(path):
    return Milvus(
        embedding_function=embeddings,
        connection_args={"uri": path},
    )


def get_vectorestore_from_docs(docs, path):
    return Milvus.from_documents(
        embedding=embeddings,
        documents=docs,
        connection_args={"uri": path},
        drop_old=True,
    )