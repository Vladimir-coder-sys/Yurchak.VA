FROM python:3.9-slim
LABEL authors="yurch"
# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /app

# Копирование зависимостей
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY . .

# Загрузка моделей spaCy
RUN python -c "import spacy; spacy.cli.download('en_core_web_sm')"
RUN python -c "import spacy; spacy.cli.download('ru_core_news_sm')"
RUN python -c "import spacy; spacy.cli.download('de_core_news_sm')"
RUN python -c "import spacy; spacy.cli.download('es_core_news_sm')"
RUN python -c "import spacy; spacy.cli.download('fr_core_news_sm')"
# Загрузка NLTK данных
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('cmudict')"

# Порт приложения
EXPOSE 5000

# Запуск приложения
CMD ["python", "app.py"]