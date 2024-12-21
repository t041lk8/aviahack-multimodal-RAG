from typing import List
import torch

from langchain_core.documents import Document
from typing import Callable, Union, Optional
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFacePipeline

from accelerate import dispatch_model, infer_auto_device_map

from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.utils import Input, Output
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

PROMPT_TEMPLATE = """
Твоя задача отвечать на вопрос, только по заданной информации. 
Тебе на вход будет подавать вопрос пользователя и контексты для ответа на этот вопрос.
Не добавляй информации от себя.
Отвечай только по предоставленной информации.

<context>
{context}
</context>

<question>
{question}
</question>
"""
rag_prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

PROMPT_TEMPLATE_CHECKPOINT = """
Твоя задача формировать чек лист для проверки настройки сельскохозяйственной техники по вопросу, только по заданной информации. 
Тебе на вход будет подаваться вопрос пользователя и контексты для ответа на этот вопрос.
Нужно по данной тебе информации написать инструкцию по пунктам.
Запрос пользователя будет в формате: "Как мне сделать ..., если у меня есть A,B,C,D и так далее."
Твой ответ должен вернусть чек-лист или план-проверки в формате: "Чтобы ..., надо сделать X, Y,Z и так далее" 
Не добавляй информации от себя. Можешь перефразировать ответы, а не отвечать 1 в 1.
Отвечай только по предоставленной информации.

<context>
{context}
</context>

<question>
{question}
</question>
$
"""

rag_prompt_checkpoint = PromptTemplate(
    template=PROMPT_TEMPLATE_CHECKPOINT, input_variables=["context", "question"]
)

EMBEDDER_MODEL_NAME = "BAAI/bge-m3"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDER_MODEL_NAME)

llm = HuggingFacePipeline.from_model_id(
    model_id="Vikhrmodels/Vikhr-Nemo-12B-Instruct-R-21-09-24",
    task="text-generation",
    model_kwargs={
        "max_memory":{0: "30GB"},
        "temperature": 0.7,
        "torch_dtype": torch.float16,  # Использование fp16
        "low_cpu_mem_usage":True,     # Оптимизация использования оперативной памяти,
        "device_map":"auto"
    },
    pipeline_kwargs = {
        "max_new_tokens":2048
    }
)

class RerankerRunnable(Runnable):
    def __init__(
        self,
        compressor: BaseDocumentCompressor,
        top_k: int = 10,
    ):
        self.compressor = compressor
        self.top_k = top_k

    def _remove_duplicates(
        self,
        retrieved_documents: List[Document],
    ):
        seen_page_contents = set()
        unique_documents = []
        for doc in retrieved_documents:
            if doc.page_content not in seen_page_contents:
                unique_documents.append(doc)
                seen_page_contents.add(doc.page_content)
        return unique_documents

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> Output:
        milvus_retrieved_doc: List[Document] = input.get("milvus_retrieved_doc")
        bm25_retrieved_doc: List[Document] = input.get("bm25_retrieved_doc")
        query: str = input.get("query")
        print(f"len(milvus_retrieved_doc) = {len(milvus_retrieved_doc)}")
        print(f"len(bm25_retrieved_doc) = {len(bm25_retrieved_doc)}")
        unique_documents = self._remove_duplicates(
            milvus_retrieved_doc + bm25_retrieved_doc
        )
        print(f"len(unique_documents) = {len(unique_documents)}")
        result = self.compressor.compress_documents(unique_documents, query)

        return result

model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-m3"
)
compressor = CrossEncoderReranker(model=model, top_n=5)
reranker = RerankerRunnable(
    compressor=compressor,
    top_k=10,
)