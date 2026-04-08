"""
RAG индекс и функции для поиска по FAQ и документам
Поддерживает загрузку FAISS индекса, поиск похожих вопросов и генерацию ответов через OpenAI
"""
import json
import os
from typing import List, Tuple, Any, Dict

import faiss
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Инициализация OpenAI клиента
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def load_index(index_path: str, meta_path: str) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    """
    Загружает FAISS индекс и метаданные.
    
    Args:
        index_path: путь к FAISS индексу (.bin)
        meta_path: путь к метаданным (.npy)
    
    Returns:
        tuple: (index, metadata)
    
    Raises:
        RuntimeError: если файлы индекса не найдены
    """
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise RuntimeError(
            "FAISS index or metadata not found. "
            "Run `python -m backend.build_index` first to build the RAG index."
        )

    index = faiss.read_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)
    return index, metadata


def search_similar(
    index: faiss.IndexFlatL2,
    metadata: np.ndarray,
    query_vec: np.ndarray,
    k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Ищет k наиболее похожих элементов в индексе.
    
    Args:
        index: FAISS индекс
        metadata: массив с метаданными
        query_vec: вектор запроса
        k: количество результатов
    
    Returns:
        list: список найденных элементов с метаданными
    """
    distances, indices = index.search(query_vec, k)
    idxs = indices[0]
    results = []
    for i in idxs:
        if 0 <= i < len(metadata):
            results.append(metadata[i])
    return results


def load_faq_data(path: str) -> List[Dict[str, str]]:
    """
    Загружает FAQ данные из JSON файла.
    
    Args:
        path: путь к JSON файлу
    
    Returns:
        list: список вопросов и ответов
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Создает эмбеддинги для текстов через OpenAI API.
    
    Args:
        texts: список текстов
    
    Returns:
        np.ndarray: массив эмбеддингов
    """
    if not client:
        raise RuntimeError("OpenAI client not initialized. Check OPENAI_API_KEY in .env")
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [d.embedding for d in response.data]
    return np.array(vectors, dtype="float32")


def generate_answer(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Генерирует ответ на вопрос пользователя используя RAG.
    
    Args:
        question: вопрос пользователя
        top_k: количество релевантных документов для контекста
    
    Returns:
        dict: словарь с полями 'answer' и 'context'
    """
    # Проверяем наличие API ключа
    if not OPENAI_API_KEY:
        return {
            'answer': 'OpenAI API ключ не настроен. Пожалуйста, добавьте OPENAI_API_KEY в файл .env',
            'context': []
        }
    
    if not client:
        return {
            'answer': 'Ошибка инициализации OpenAI клиента. Проверьте API ключ.',
            'context': []
        }
    
    # Пути к индексам (относительно текущего файла)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
    META_PATH = os.path.join(DATA_DIR, "faqs_metadata.npy")
    
    # Загружаем индекс
    try:
        index, metadata = load_index(INDEX_PATH, META_PATH)
    except RuntimeError as e:
        return {
            'answer': 'База знаний ещё не построена. Пожалуйста, подождите, администратор настраивает систему.',
            'context': [],
            'error': str(e)
        }
    
    # Создаём эмбеддинг вопроса
    try:
        query_vec = embed_texts([question])
    except Exception as e:
        return {
            'answer': 'Ошибка при обработке вопроса. Попробуйте позже.',
            'context': [],
            'error': str(e)
        }
    
    # Ищем похожие вопросы
    similar_items = search_similar(index, metadata, query_vec, k=top_k)
    
    if not similar_items:
        return {
            'answer': 'Не нашёл подходящего ответа в базе знаний. Пожалуйста, свяжитесь с нами через форму обратной связи.',
            'context': []
        }
    
    # Формируем контекст для OpenAI
    context_text = "\n\n".join(
        [f"Вопрос: {item['question']}\nОтвет: {item['answer']}" for item in similar_items]
    )
    
    # Промпт для OpenAI
    system_prompt = (
        "Ты FAQ-ассистент компании. Отвечай кратко, дружелюбно и по делу на русском языке. "
        "Используй предоставленный контекст с вопросами и ответами. "
        "Если в контексте нет точной информации для ответа на вопрос, честно скажи об этом "
        "и предложи обратиться в поддержку через форму обратной связи на сайте. "
        "Не придумывай ответы, которых нет в контексте."
    )
    
    user_prompt = f"""Вопрос пользователя: {question}

Контекст из базы знаний (наиболее релевантные FAQ):
{context_text}

Дай краткий и полезный ответ на русском языке, основываясь ТОЛЬКО на предоставленном контексте."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # используем более доступную модель
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        # Если OpenAI не отвечает, используем fallback ответ из найденных FAQ
        answer = similar_items[0]['answer'] if similar_items else \
                 'Извините, возникла техническая ошибка. Пожалуйста, свяжитесь с нами через форму обратной связи.'
    
    return {
        'answer': answer,
        'context': [dict(item) for item in similar_items]
    }


def get_answer_from_fallback(question: str, similar_items: List[Dict[str, Any]]) -> str:
    """
    Fallback функция для получения ответа без OpenAI (просто возвращает первый найденный ответ).
    
    Args:
        question: вопрос пользователя
        similar_items: найденные похожие элементы
    
    Returns:
        str: ответ
    """
    if similar_items:
        return similar_items[0]['answer']
    return "Извините, не удалось найти ответ на ваш вопрос. Пожалуйста, свяжитесь с нами через форму обратной связи."