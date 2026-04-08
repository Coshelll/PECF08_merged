# Веб-сайт с AI-ассистентом (RAG система)

Полнофункциональный веб-сайт на Python Flask с формой обратной связи, кейсами, админ-панелью и **интеллектуальным FAQ-ассистентом** на базе RAG (Retrieval-Augmented Generation).

## 🚀 Возможности

- **Главная страница** с hero-блоком и описанием
- **5 кейсов** с детальными описаниями
- **Форма обратной связи** с валидацией и сохранением в БД
- **Админ-панель** с авторизацией для управления заявками
- **🤖 RAG чат-бот** с поиском по базе знаний (FAISS + OpenAI)
- **Современный дизайн** с Bootstrap 5
- **Адаптивная верстка** для всех устройств

## 🧠 RAG Чат-ассистент

Интеллектуальный помощник отвечает на вопросы пользователей, используя:
- Базу знаний из FAQ (JSON) и текстовых документов
- Векторный поиск через FAISS для быстрого нахождения релевантных ответов
- OpenAI GPT для генерации точных, контекстных ответов на русском языке
- Готовый чат-виджет, встроенный прямо на сайт

## 📋 Требования

- Python 3.8 или выше
- OpenAI API ключ (для работы чат-бота)

## 🛠️ Установка

**1. Клонируйте репозиторий:**
```bash
git clone https://github.com/Coshelll/PECF08_merged.git
cd PECF08_merged
2. Создайте виртуальное окружение:

bash
python -m venv venv
3. Активируйте виртуальное окружение:

Windows:

bash
venv\Scripts\activate
Linux/Mac:

bash
source venv/bin/activate
4. Установите зависимости:

bash
pip install -r requirements.txt
5. Настройте переменные окружения:

Создайте файл .env в корне проекта:

env
OPENAI_API_KEY=ваш_ключ_openai
SECRET_KEY=ваш_секретный_ключ
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
6. Постройте RAG индекс (базу знаний):

bash
python -m backend.build_index
▶️ Запуск
bash
python app.py
Приложение будет доступно по адресу: http://localhost:5000

🔐 Доступ к админ-панели
URL: http://localhost:5000/admin/login

Логин по умолчанию: admin

Пароль по умолчанию: admin123

⚠️ В продакшене обязательно измените учетные данные!

📁 Структура проекта
text
PECF08_merged/
├── app.py                 # Главный файл приложения
├── config.py              # Конфигурация
├── requirements.txt       # Зависимости
├── .env                   # Переменные окружения
├── backend/               # RAG модуль
│   ├── build_index.py     
│   └── rag_index.py       
├── data/                  # Данные для RAG
│   ├── faqs.json          
│   └── *.txt              
├── templates/             # HTML шаблоны
└── static/                # CSS, JS, изображения
🎨 Технологии
Backend: Flask, OpenAI API, FAISS, SQLite

Frontend: Bootstrap 5, Bootstrap Icons, JavaScript

🔧 Настройка RAG базы знаний
Добавление своих FAQ — отредактируйте data/faqs.json:

json
[
  {"question": "Ваш вопрос", "answer": "Ваш ответ"}
]
Добавление текстовых документов — положите .txt файлы в папку data/

Перестроение индекса после изменений:

bash
python -m backend.build_index
🐛 Решение проблем
Ошибка OpenAI API (региональная блокировка)

Используйте VPN

Или замените на локальную модель sentence-transformers

Чат-бот не отвечает

Проверьте наличие .env с OPENAI_API_KEY

Перестройте индекс: python -m backend.build_index

👨‍💻 Автор
Coshelll

Приятного использования! 🎉