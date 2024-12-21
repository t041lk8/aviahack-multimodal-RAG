import requests
import time
from src.indexer import get_processed_docs, load_vectorestore_from_file
from src.models import (
    rag_prompt,
    llm,
    reranker,
    rag_prompt_checkpoint
)
from langchain_community.retrievers import BM25Retriever

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

import warnings

from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)

GET_URL = "http://176.108.255.29:5555/get_message"
POST_URL = "http://176.108.255.29:5555/process"

def get_message():
    """
    Делает GET-запрос, возвращает dict c данными сообщения или None.
    """
    try:
        response = requests.get(GET_URL, timeout=10)
        if response.status_code == 200:
            message_data = response.json()
            # Если вернулся JSON вида {"id": ..., "user_id": ..., "text": ...},
            # то возвращаем его. Иначе, по своему желанию (None или обработка).
            if "id" in message_data and "text" in message_data:
                return message_data
        else:
            print(f"GET вернул статус: {response.status_code}, текст: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при GET-запросе: {e}")
        return None

def send_reply(process_data):
    """
    Делаем POST-запрос для отправки ответа.
    process_data — это dict вида:
      {
        "message_id": <int>,
        "reply_text": <str>,
        "user_id": <int>
      }
    """
    try:
        response = requests.post(POST_URL, json=process_data, timeout=10)
        if response.status_code != 200:
            print(f"POST вернул статус: {response.status_code}, текст: {response.text}")
        else:
            print("Ответ успешно отправлен:", process_data)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при POST-запросе: {e}")



docs = get_processed_docs()
vectorstore = load_vectorestore_from_file("milvus_demo.db")
milvus_retriever = vectorstore.as_retriever()
bm25_retriever = BM25Retriever.from_documents(docs)

hybrid_and_rerank_retriever = {
    "milvus_retrieved_doc": milvus_retriever,
    "bm25_retrieved_doc": bm25_retriever,
    "query": RunnablePassthrough(),
} | reranker

hybrid_and_rerank_chain = (
    {
        "context": hybrid_and_rerank_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

hybrid_and_rerank_chain_checkpoint = (
    {
        "context": hybrid_and_rerank_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt_checkpoint
    | llm
    | StrOutputParser()
)

sub_query_chain = (
    {
        "context": sub_query_retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
).with_types(input_type=Question)

def process_message(message):
    """
    Здесь вызывается ваш RAG LLM. На вход приходит dict:
      {
        "id": 247,
        "user_id": 447657726,
        "text": "fwefwe"
      }
    Нужно вернуть dict для /process:
      {
        "message_id": <int>,
        "reply_text": <str>,
        "user_id": <int>
      }
    """
    
    # 1. Получаем текст вопроса
    user_text = message["text"]
    
    if message['mode'] == "search":
        hybrid_and_rerank_result = hybrid_and_rerank_chain.invoke(user_text)
        answer = hybrid_and_rerank_result.split("$")[1]
    elif message['mode'] == "checklist":
        hybrid_and_rerank_checkpoint_result = hybrid_and_rerank_chain_checkpoint.invoke(user_text)
        answer = hybrid_and_rerank_checkpoint_result.split("$")[1]
    else:
        answer="Error! No such mode!"
    
    # 3. Сформируем данные для отправки
    result = {
        "message_id": message["id"],
        "reply_text": answer,
        "user_id": message["user_id"]
    }
    return result

def main_loop():
    """
    Основной цикл: бесконечно забираем сообщения, обрабатываем, отправляем ответ.
    """
    while True:
        message_data = get_message()

        # Если нет нового сообщения, ждем немного и пробуем снова
        if not message_data:
            time.sleep(5)
            continue
        
        # Обрабатываем сообщение через LLM RAG
        reply_data = process_message(message_data)
        
        # Отправляем ответ
        send_reply(reply_data)
        
        # Небольшая пауза, чтобы не захлестывать сервер запросами.
        # При желании можно убрать или изменить время.
        time.sleep(1)

if __name__ == "__main__":
    query = "Как настроить время?"

    hybrid_and_rerank_result = hybrid_and_rerank_chain.invoke(query)
    answer = hybrid_and_rerank_result.split("$")[1]
    print(answer)