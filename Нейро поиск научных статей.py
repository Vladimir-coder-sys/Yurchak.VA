import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
import pickle
import tempfile
import textwrap
import uuid
import json
import re
import time
from datetime import datetime
from collections import Counter, defaultdict
from io import StringIO, BytesIO

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
import plotly.graph_objects as go
import plotly.express as px
import base64
import networkx as nx
import nltk
import spacy
from spacy.matcher import Matcher
from langdetect import detect, DetectorFactory
import PyPDF2
import docx2txt
import openpyxl
import cv2
import easyocr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
import torch
from rank_bm25 import BM25Okapi

from flask import Flask, request, render_template_string, redirect, url_for, session
from Bio import Entrez
from Bio import Medline
import pullenti
from pullenti.ner import Referent

# Убедимся в воспроизводимости результатов
DetectorFactory.seed = 0
np.random.seed(42)
torch.manual_seed(42)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка NLP ресурсов
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('cmudict', quiet=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")
Entrez.email = "yurchak777x@gmail.com"

# Инициализация Pullenti SDK
try:
    pullenti.ProcessorService.initialize()
    logger.info("Pullenti SDK initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pullenti SDK: {str(e)}")

# Инициализация моделей
SPACY_MODELS = {
    'en': 'en_core_web_md',
    'ru': 'ru_core_news_md',
    'es': 'es_core_news_md',
    'fr': 'fr_core_news_md',
    'de': 'de_core_news_md',
    'zh': 'zh_core_web_md'
}
nlp_models = {}
for lang, model_name in SPACY_MODELS.items():
    try:
        nlp = spacy.load(model_name)
        nlp.max_length = 5000000
        nlp_models[lang] = nlp
        logger.info(f"Loaded spaCy model for {lang}")
    except Exception as e:
        logger.warning(f"Could not load spaCy model for {lang}: {str(e)}")
        nlp_models[lang] = None

# Инициализация моделей трансформеров
MODEL_CACHE = {}
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

    # Базовые модели
    MODEL_CACHE['t5_tokenizer'] = T5Tokenizer.from_pretrained("t5-small")
    MODEL_CACHE['t5_model'] = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Попытка загрузки SentenceTransformer с обработкой ошибок
    try:
        from sentence_transformers import SentenceTransformer

        MODEL_CACHE['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SentenceTransformer model loaded successfully")
    except ImportError as ie:
        logger.warning(f"SentenceTransformer not available: {str(ie)}")
        MODEL_CACHE['sentence_model'] = None
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer: {str(e)}")
        MODEL_CACHE['sentence_model'] = None

    # Модель для генерации
    MODEL_CACHE['kag_tokenizer'] = AutoTokenizer.from_pretrained("google/flan-t5-large")
    MODEL_CACHE['kag_model'] = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

    logger.info("Transformer models loaded successfully")
except Exception as e:
    logger.error(f"Error loading transformer models: {str(e)}")

# Конфигурация и кэширование
TERM_CACHE_FILE = "term_cache.pkl"
term_cache = {}
if os.path.exists(TERM_CACHE_FILE):
    try:
        with open(TERM_CACHE_FILE, 'rb') as f:
            term_cache = pickle.load(f)
        logger.info(f"Loaded term cache with {len(term_cache)} entries")
    except Exception as e:
        logger.error(f"Error loading term cache: {str(e)}")
        term_cache = {}

# API ключи
CORE_API_KEY = os.getenv("CORE_API_KEY", "3qj6Ggun9KY1AJPOlvLRHpmVUxkZEeCo")
UMLS_API_KEY = os.getenv("UMLS_API_KEY", "5266ecdf-b5ef-4f6c-922b-55c224f950b9")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY", "dk_bIGNaCGH-qw8PvsZu9coMcu-CGXC7")
ABBYY_APP_ID = os.getenv("ABBYY_APP_ID", "51a7f7a1-ccba-412a-b831-59a630780da0")
ABBYY_PASSWORD = os.getenv("ABBYY_PASSWORD", "GETE#911*Uc13")

# Оптимизированный список медицинских терминологических источников
PRIO_SABS = [
    "SNOMEDCT_US", "MSH", "ICD10CM", "RXNORM", "CPT", "LOINC",
    "ICD10PCS", "LNC", "NCI", "NDFRT", "MTH", "NCI_CTCAE",
    "NCI_FDA", "NCI_NCI-GLOSS", "SCTSPA", "VANDF"
]

# Медицинские паттерны
MEDICAL_PATTERNS = {
    'en': [
        [{"ENT_TYPE": "DISEASE"}],
        [{"ENT_TYPE": "SYMPTOM"}],
        [{"ENT_TYPE": "CHEM"}],
        [{"ENT_TYPE": "ANAT"}],
        [{"ENT_TYPE": "GENE"}],
        [{"ENT_TYPE": "PROTEIN"}],
        [{"ENT_TYPE": "PROCEDURE"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "ADJ"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["treatment", "therapy", "diagnosis", "prognosis"]}}],
        [{"LOWER": {"IN": ["syndrome", "disorder", "disease", "pathology"]}}],
        [{"LOWER": {"IN": ["cancer", "tumor", "neoplasm", "carcinoma"]}}],
        [{"LOWER": {"IN": ["infection", "bacterial", "viral", "fungal"]}}],
        [{"LOWER": {"IN": ["surgery", "procedure", "operation", "intervention"]}}]
    ],
    'ru': [
        [{"ENT_TYPE": "DISEASE"}],
        [{"ENT_TYPE": "SYMPTOM"}],
        [{"ENT_TYPE": "CHEM"}],
        [{"ENT_TYPE": "ANAT"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "ADJ"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["лечение", "терапия", "диагностика", "прогноз"]}}],
        [{"LOWER": {"IN": ["синдром", "заболевание", "нарушение", "патология"]}}],
        [{"LOWER": {"IN": ["рак", "опухоль", "новообразование", "карцинома"]}}],
        [{"LOWER": {"IN": ["инфекция", "бактериальная", "вирусная", " грибковая"]}}],
        [{"LOWER": {"IN": ["операция", "процедура", "вмешательство", "хирургия"]}}]
    ],
    # Шаблоны для других языков
    'es': [
        [{"ENT_TYPE": "DISEASE"}],
        [{"ENT_TYPE": "SYMPTOM"}],
        [{"ENT_TYPE": "CHEM"}],
        [{"ENT_TYPE": "ANAT"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "ADJ"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["tratamiento", "terapia", "diagnóstico", "pronóstico"]}}],
        [{"LOWER": {"IN": ["síndrome", "trastorno", "enfermedad", "patología"]}}],
        [{"LOWER": {"IN": ["cáncer", "tumor", "neoplasia", "carcinoma"]}}],
        [{"LOWER": {"IN": ["infección", "bacteriana", "viral", "fúngica"]}}],
        [{"LOWER": {"IN": ["cirugía", "procedimiento", "intervención", "operación"]}}]
    ],
    'fr': [
        [{"ENT_TYPE": "DISEASE"}],
        [{"ENT_TYPE": "SYMPTOM"}],
        [{"ENT_TYPE": "CHEM"}],
        [{"ENT_TYPE": "ANAT"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "ADJ"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["traitement", "thérapie", "diagnostic", "pronostic"]}}],
        [{"LOWER": {"IN": ["syndrome", "trouble", "maladie", "pathologie"]}}],
        [{"LOWER": {"IN": ["cancer", "tumeur", "néoplasie", "carcinome"]}}],
        [{"LOWER": {"IN": ["infection", "bactérienne", "virale", "fongique"]}}],
        [{"LOWER": {"IN": ["chirurgie", "procédure", "intervention", "opération"]}}]
    ],
    'de': [
        [{"ENT_TYPE": "DISEASE"}],
        [{"ENT_TYPE": "SYMPTOM"}],
        [{"ENT_TYPE": "CHEM"}],
        [{"ENT_TYPE": "ANAT"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "ADJ"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["behandlung", "therapie", "diagnose", "prognose"]}}],
        [{"LOWER": {"IN": ["syndrom", "erkrankung", "störung", "pathologie"]}}],
        [{"LOWER": {"IN": ["krebs", "tumor", "neoplasie", "karzinom"]}}],
        [{"LOWER": {"IN": ["infektion", "bakteriell", "viral", "pilzartig"]}}],
        [{"LOWER": {"IN": ["chirurgie", "verfahren", "eingriff", "operation"]}}]
    ],
    'zh': [
        [{"ENT_TYPE": "DISEASE"}],
        [{"ENT_TYPE": "SYMPTOM"}],
        [{"ENT_TYPE": "CHEM"}],
        [{"ENT_TYPE": "ANAT"}],
        [{"POS": "NOUN"}, {"POS": "NOUN"}],
        [{"POS": "ADJ"}, {"POS": "NOUN"}],
        [{"POS": "NOUN"}, {"POS": "ADP"}, {"POS": "NOUN"}],
        [{"LOWER": {"IN": ["治疗", "疗法", "诊断", "预后"]}}],
        [{"LOWER": {"IN": ["综合征", "疾病", "障碍", "病理"]}}],
        [{"LOWER": {"IN": ["癌症", "肿瘤", "新生物", "癌"]}}],
        [{"LOWER": {"IN": ["感染", "细菌", "病毒", "真菌"]}}],
        [{"LOWER": {"IN": ["手术", "程序", "干预", "操作"]}}]
    ]
}

# Медицинские сущности
MEDICAL_ENTITIES = {
    'diseases': ["DISEASE", "PATHOLOGY"],
    'drugs': ["CHEM", "PHARM"],
    'procedures': ["PROCEDURE", "TREATMENT"],
    'symptoms': ["SYMPTOM", "MANIFESTATION"],
    'anatomy': ["ANAT", "BODY_PART"],
    'genes': ["GENE", "GENETIC"],
    'proteins': ["PROTEIN", "ENZYME"]
}

# Медицинские темы
MEDICAL_TOPICS = {
    "Онкология": ["рак", "опухоль", "химиотерапия", "карцинома", "метастаз"],
    "Кардиология": ["сердце", "инфаркт", "гипертензия", "аритмия", "стент"],
    "Неврология": ["мозг", "инсульт", "болезнь Альцгеймера", "Паркинсон", "нейроны"],
    "Инфекционные болезни": ["инфекция", "вирус", "бактерия", "антибиотик", "вакцина"],
    "Эндокринология": ["диабет", "гормон", "щитовидная железа", "инсулин", "метаболизм"],
    "Гастроэнтерология": ["желудок", "кишечник", "гепатит", "язва", "пищеварение"],
    "Педиатрия": ["дети", "новорожденные", "развитие", "вакцинация", "врожденный"],
    "Хирургия": ["операция", "анестезия", "трансплантация", "лапароскопия", "шов"],
    "Генетика": ["ДНК", "ген", "мутация", "наследственность", "геном"],
    "Иммунология": ["иммунитет", "антитело", "аллергия", "аутоиммунный", "вакцина"]
}

# HTML шаблоны
INDEX_HTML = """
<!DOCTYPE html><html lang="ru"><head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Нейро поиск исследовательских статей</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .gradient-bg {
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            min-height: 100vh;
            padding: 2rem 0;
        }
        .card {
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            border: none;
        }
        .logo {
            max-width: 220px;
            margin: 0 auto 1.5rem;
            display: block;
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #1a2980;
        }
        .upload-section {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 2rem;
        }
        .btn-neuro {
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            border: none;
            color: white;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .btn-neuro:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .search-form .form-control {
            border: 2px solid #e9ecef;
            padding: 0.75rem;
            border-radius: 10px;
        }
        .search-form .form-control:focus {
            border-color: #1a2980;
            box-shadow: 0 0 0 0.25rem rgba(26, 41, 128, 0.25);
        }
        .feature-card {
            transition: transform 0.3s ease;
            height: 100%;
        }
        .feature-card:hover {
            transform: translateY(-10px);
        }
        .stats-counter {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .topic-selector {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
        }
        .topic-btn {
            padding: 5px 12px;
            border-radius: 20px;
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            cursor: pointer;
            transition: all 0.2s;
        }
        .topic-btn:hover, .topic-btn.active {
            background: #1a2980;
            color: white;
        }
    </style>
</head>
<body class="gradient-bg">
    <div class="container py-5">
        <div class="text-center mb-5">
            <img src="https://upload.wikimedia.org/wikipedia/ru/8/86/Rosnou_logo.png" alt="РосНОУ логотип" class="logo">
            <h1 class="text-white mb-3">Нейро поиск исследовательских статей</h1>
            <p class="lead text-white mb-4">Используйте искусственный интеллект для семантического поиска в научных базах знаний</p>
        </div>

        <div class="row justify-content-center">
            <div class="col-lg-8">
                <div class="card">
                    <div class="card-body p-4">
                        <form action="/search" method="post" class="search-form">
                            <div class="mb-3">
                                <input type="text" name="query" class="form-control form-control-lg" 
                                       placeholder="Введите поисковый запрос" required>
                            </div>

                            <div class="row g-3 mb-3">
                                <div class="col-md-4">
                                    <select name="db" class="form-select">
                                        <option value="pubmed">PubMed</option>
                                        <option value="core">CORE</option>
                                        <option value="combined">Combined</option>
                                    </select>
                                </div>
                                <div class="col-md-4">
                                    <input type="number" name="year_start" class="form-control" 
                                           placeholder="Год начала" min="1900" max="2025" value="2018">
                                </div>
                                <div class="col-md-4">
                                    <input type="number" name="year_end" class="form-control" 
                                           placeholder="Год окончания" min="1900" max="2025" value="2023">
                                </div>
                            </div>

                            <div class="mb-3">
                                <input type="text" name="exclude_terms" class="form-control" 
                                       placeholder="Исключаемые термины (через запятую)">
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Выберите тематику (опционально):</label>
                                <div class="topic-selector">
                                    {% for topic in medical_topics %}
                                        <div class="topic-btn" onclick="toggleTopic(this)">{{ topic }}</div>
                                    {% endfor %}
                                </div>
                                <input type="hidden" name="selected_topics" id="selectedTopics">
                            </div>

                            <div class="d-grid">
                                <button type="submit" class="btn btn-neuro">Поиск</button>
                            </div>
                        </form>

                        <div class="upload-section mt-4">
                            <h5 class="mb-3">Или загрузите документ для анализа</h5>
                            <form action="/upload" method="post" enctype="multipart/form-data">
                                <div class="input-group">
                                    <input type="file" name="file" class="form-control" 
                                           accept=".pdf,.docx,.xlsx,.png,.jpg,.tiff,.txt">
                                    <button class="btn btn-outline-primary" type="submit">Анализировать</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-5">
            <div class="col-md-4 mb-4">
                <div class="card feature-card h-100">
                    <div class="card-body text-center p-4">
                        <div class="feature-icon">🔍</div>
                        <h4 class="card-title">Семантический поиск</h4>
                        <p class="card-text">Поиск по смыслу, а не только по ключевым словам, с использованием нейросетей</p>
                    </div>
                </div>
            </div>

            <div class="col-md-4 mb-4">
                <div class="card feature-card h-100">
                    <div class="card-body text-center p-4">
                        <div class="feature-icon">🧠</div>
                        <h4 class="card-title">Анализ связей</h4>
                        <p class="card-text">Выявление скрытых взаимосвязей между исследованиями и терминами</p>
                    </div>
                </div>
            </div>

            <div class="col-md-4 mb-4">
                <div class="card feature-card h-100">
                    <div class="card-body text-center p-4">
                        <div class="feature-icon">📊</div>
                        <h4 class="card-title">Визуализация данных</h4>
                        <p class="card-text">Интерактивные графики и диаграммы для анализа результатов</p>
                    </div>
                </div>
            </div>
        </div>

        <div class="row mt-5 text-center">
            <div class="col-md-3 mb-3">
                <div class="card py-3">
                    <div class="stats-counter">100K+</div>
                    <p class="mb-0">Статей в базе</p>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card py-3">
                    <div class="stats-counter">50+</div>
                    <p class="mb-0">Медицинских терминологий</p>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card py-3">
                    <div class="stats-counter">10K+</div>
                    <p class="mb-0">Пользователей</p>
                </div>
            </div>
            <div class="col-md-3 mb-3">
                <div class="card py-3">
                    <div class="stats-counter">24/7</div>
                    <p class="mb-0">Доступность</p>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function toggleTopic(element) {
            element.classList.toggle('active');
            updateSelectedTopics();
        }

        function updateSelectedTopics() {
            const selected = Array.from(document.querySelectorAll('.topic-btn.active'))
                .map(el => el.textContent);
            document.getElementById('selectedTopics').value = selected.join(',');
        }

        // Инициализация тематик
        const medicalTopics = {{ medical_topics|tojson }};
        const container = document.querySelector('.topic-selector');
        medicalTopics.forEach(topic => {
            const btn = document.createElement('div');
            btn.className = 'topic-btn';
            btn.textContent = topic;
            btn.onclick = () => toggleTopic(btn);
            container.appendChild(btn);
        });
    </script>
</body>
</html>
"""

RESULTS_HTML = """
<!DOCTYPE html><html lang="ru"><head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Результаты поиска</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
            min-height: 100vh;
            padding: 2rem 0;
        }
        .card {
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: 1.5rem;
            border: none;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        .header-card {
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            color: white;
            border-radius: 12px;
        }
        .entity-badge {
            background: #e3f2fd;
            color: #1565c0;
            padding: 0.3rem 0.6rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 0.2rem;
            display: inline-block;
            border: 1px solid #bbdefb;
        }
        .topic-badge {
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            display: inline-block;
            margin-bottom: 0.5rem;
        }
        .viz-card {
            height: 100%;
            overflow: hidden;
        }
        .viz-container {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
        }
        .semantic-term {
            background: #e3f2fd;
            border-bottom: 2px dashed #64b5f6;
            cursor: help;
            padding: 0 2px;
            border-radius: 3px;
        }
        .nav-tabs .nav-link.active {
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            color: white !important;
            border: none;
            font-weight: 600;
        }
        .nav-tabs .nav-link {
            color: #1a2980;
            font-weight: 500;
        }
        .insight-card {
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 1.2rem;
            border-radius: 0 8px 8px 0;
            margin: 1.5rem 0;
        }
        .btn-back {
            background: #1a2980;
            color: white;
            padding: 0.6rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .btn-back:hover {
            background: #0d1a5a;
            transform: translateY(-2px);
        }
        .keyword-badge {
            background: #ffecb3;
            color: #5d4037;
            padding: 0.3rem 0.6rem;
            border-radius: 20px;
            font-size: 0.85rem;
            margin: 0.2rem;
            display: inline-block;
        }
        .document-card {
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .document-card:hover {
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        }
        .topic-tag {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            background: #e3f2fd;
            margin: 0.2rem;
            font-size: 0.85rem;
        }
        .dashboard-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-top: 30px;
        }
        .dashboard-item {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .dashboard-title {
            font-size: 18px;
            margin-bottom: 15px;
            color: #1a2980;
            font-weight: bold;
        }
        .relation-tag {
            background: #e1bee7;
            color: #4a148c;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 0 3px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header-card p-4 mb-4">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h1 class="mb-1">Результаты поиска</h1>
                    <p class="mb-0">База: {{ db|upper }} | Период: {{ year_start }} - {{ year_end }} | Найдено: {{ results|length }} статей</p>
                    {% if selected_topics %}
                        <p class="mb-0">Тематики: {{ selected_topics }}</p>
                    {% endif %}
                </div>
                <a href="/" class="btn btn-light">Новый поиск</a>
            </div>
        </div>

        {% if not results or results[0].get('TI') == 'Результаты не найдены' %}
            <div class="card">
                <div class="card-body text-center p-5">
                    <h2 class="text-muted">Результаты не найдены</h2>
                    <p>Попробуйте изменить параметры поиска или использовать другие ключевые слова</p>
                    <a href="/" class="btn btn-primary mt-3">Вернуться к поиску</a>
                </div>
            </div>
        {% else %}
            <!-- Аналитические инсайты -->
            {% if insights %}
            <div class="insight-card">
                <h4><i class="bi bi-lightbulb"></i> Ключевые инсайты</h4>
                <p>{{ insights }}</p>
            </div>
            {% endif %}

            <!-- Результаты поиска -->
            <div class="row">
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-body">
                            <h4 class="card-title mb-4">Найденные публикации</h4>

                            <ul class="nav nav-tabs mb-3" id="resultTabs" role="tablist">
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link active" id="list-tab" data-bs-toggle="tab" 
                                            data-bs-target="#list" type="button" role="tab">Список</button>
                                </li>
                                <li class="nav-item" role="presentation">
                                    <button class="nav-link" id="cards-tab" data-bs-toggle="tab" 
                                            data-bs-target="#cards" type="button" role="tab">Карточки</button>
                                </li>
                            </ul>

                            <div class="tab-content" id="resultTabsContent">
                                <!-- Список результатов -->
                                <div class="tab-pane fade show active" id="list" role="tabpanel">
                                    <div class="table-responsive">
                                        <table class="table table-hover">
                                            <thead>
                                                <tr>
                                                    <th>Название</th>
                                                    <th>Год</th>
                                                    <th>Источник</th>
                                                    <th>Тема</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {% for result in results %}
                                                <tr onclick="showDetails('{{ loop.index0 }}')" style="cursor: pointer;">
                                                    <td>{{ result.get('TI', 'No title')|truncate(60) }}</td>
                                                    <td>{{ result.get('DP', 'N/A') }}</td>
                                                    <td>{{ result.get('Source', 'N/A') }}</td>
                                                    <td><span class="topic-badge">{{ result.get('Topic', 'No topic')|truncate(20) }}</span></td>
                                                </tr>
                                                {% endfor %}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>

                                <!-- Карточки результатов -->
                                <div class="tab-pane fade" id="cards" role="tabpanel">
                                    <div class="row">
                                        {% for result in results %}
                                        <div class="col-md-6 mb-3">
                                            <div class="card document-card" onclick="showDetails('{{ loop.index0 }}')">
                                                <div class="card-body">
                                                    <span class="topic-badge">{{ result.get('Topic', 'No topic')|truncate(20) }}</span>
                                                    <h5 class="card-title">{{ result.get('TI', 'No title')|truncate(60) }}</h5>
                                                    <p class="card-text text-muted small">
                                                        {{ result.get('DP', 'N/A') }} | {{ result.get('Source', 'N/A') }}
                                                    </p>
                                                    <div class="mb-2">
                                                        {% for kw in result.get('KW', [])[:3] %}
                                                            <span class="keyword-badge">{{ kw }}</span>
                                                        {% endfor %}
                                                    </div>
                                                    <p class="card-text small">
                                                        {{ result.get('AB', 'No abstract')|striptags|truncate(150) }}
                                                    </p>
                                                    {% if result.get('topic_tags') %}
                                                    <div class="mt-2">
                                                        {% for tag in result.get('topic_tags')[:3] %}
                                                            <span class="topic-tag">{{ tag }}</span>
                                                        {% endfor %}
                                                    </div>
                                                    {% endif %}
                                                </div>
                                            </div>
                                        </div>
                                        {% endfor %}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Детали документа -->
                <div class="col-md-4">
                    <div class="card sticky-top" style="top: 20px;">
                        <div class="card-body">
                            <h4 class="card-title mb-4">Детали публикации</h4>
                            <div id="document-details">
                                <p class="text-center text-muted">Выберите публикацию для просмотра деталей</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Визуализации -->
            <div class="viz-container mt-4">
                <h4 class="mb-4">Анализ результатов</h4>

                <div class="row">
                    {% if cluster_plot %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Тематические кластеры</h5>
                                <div id="cluster_plot">{{ cluster_plot|safe }}</div>
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if year_plot %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Распределение по годам</h5>
                                <img src="data:image/png;base64,{{ year_plot }}" class="img-fluid" alt="Year distribution">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if wordcloud %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Облако терминов</h5>
                                <img src="data:image/png;base64,{{ wordcloud }}" class="img-fluid" alt="Word cloud">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if network_plot %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Семантическая сеть</h5>
                                <div id="network_plot">{{ network_plot|safe }}</div>
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if topics_distribution %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Распределение тематик</h5>
                                <img src="data:image/png;base64,{{ topics_distribution }}" class="img-fluid" alt="Topics distribution">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if venn_diagram %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Диаграмма Венна по темам</h5>
                                <img src="data:image/png;base64,{{ venn_diagram }}" class="img-fluid" alt="Venn diagram">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if topic_trends %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Динамика тематик</h5>
                                <img src="data:image/png;base64,{{ topic_trends }}" class="img-fluid" alt="Topic trends">
                            </div>
                        </div>
                    </div>
                    {% endif %}

                    {% if term_frequency %}
                    <div class="col-md-6 mb-4">
                        <div class="card viz-card">
                            <div class="card-body">
                                <h5 class="card-title">Частотность терминов</h5>
                                <img src="data:image/png;base64,{{ term_frequency }}" class="img-fluid" alt="Term frequency">
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>
            </div>

            <!-- Аналитический дашборд -->
            {% if dendrogram or semantic_graph or timeline or correlation_matrix %}
            <div class="viz-container mt-5">
                <h4 class="mb-4">Аналитический дашборд</h4>

                <div class="dashboard-container">
                    {% if dendrogram %}
                    <div class="dashboard-item">
                        <div class="dashboard-title">Кластеризация терминов</div>
                        <img src="data:image/png;base64,{{ dendrogram }}" class="img-fluid">
                    </div>
                    {% endif %}

                    {% if semantic_graph %}
                    <div class="dashboard-item">
                        <div class="dashboard-title">Семантическая сеть</div>
                        <div id="semantic_graph">{{ semantic_graph|safe }}</div>
                    </div>
                    {% endif %}

                    {% if timeline %}
                    <div class="dashboard-item">
                        <div class="dashboard-title">Динамика публикаций</div>
                        <img src="data:image/png;base64,{{ timeline }}" class="img-fluid">
                    </div>
                    {% endif %}

                    {% if correlation_matrix %}
                    <div class="dashboard-item">
                        <div class="dashboard-title">Корреляция терминов</div>
                        <img src="data:image/png;base64,{{ correlation_matrix }}" class="img-fluid">
                    </div>
                    {% endif %}
                </div>
            </div>
            {% endif %}

            <!-- Скрытые данные для JS -->
            <div id="results-data" style="display: none;">
                {{ results|tojson }}
            </div>
        {% endif %}

        <div class="text-center mt-4">
            <a href="/" class="btn-back">← Вернуться к поиску</a>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function showDetails(index) {
            const results = JSON.parse(document.getElementById('results-data').textContent);
            const result = results[index];

            let authors = result.AU ? result.AU.join(', ') : 'N/A';
            let keywords = result.KW ? result.KW.map(kw => `<span class="keyword-badge">${kw}</span>`).join(' ') : 'Нет ключевых слов';

            let topicTags = '';
            if (result.topic_tags && result.topic_tags.length > 0) {
                topicTags = `<div class="mb-2">
                    <strong>Тематические теги:</strong> 
                    ${result.topic_tags.map(tag => `<span class="topic-tag">${tag}</span>`).join('')}
                </div>`;
            }

            let entitiesHtml = '';
            if (result.entities) {
                for (const [type, items] of Object.entries(result.entities)) {
                    if (items.length > 0 && type !== 'semantic_features') {
                        entitiesHtml += `<div class="mb-2">
                            <strong>${type}:</strong> 
                            ${items.map(item => `<span class="entity-badge">${item}</span>`).join('')}
                        </div>`;
                    }
                }
            }

            let semanticFeatures = '';
            if (result.entities && result.entities.semantic_features) {
                const sem = result.entities.semantic_features;

                if (sem.synonyms && Object.keys(sem.synonyms).length > 0) {
                    semanticFeatures += `<div class="mb-3"><strong>Синонимы:</strong>`;
                    for (const [term, syns] of Object.entries(sem.synonyms)) {
                        semanticFeatures += `<div class="mt-1">${term}: ${syns.map(s => `<span class="relation-tag">${s}</span>`).join(' ')}</div>`;
                    }
                    semanticFeatures += `</div>`;
                }

                if (sem.antonyms && Object.keys(sem.antonyms).length > 0) {
                    semanticFeatures += `<div class="mb-3"><strong>Антонимы:</strong>`;
                    for (const [term, ants] of Object.entries(sem.antonyms)) {
                        semanticFeatures += `<div class="mt-1">${term}: ${ants.map(a => `<span class="relation-tag">${a}</span>`).join(' ')}</div>`;
                    }
                    semanticFeatures += `</div>`;
                }

                if (sem.semantic_graph && sem.semantic_graph.length > 0) {
                    semanticFeatures += `<div class="mb-3"><strong>Семантические связи:</strong>`;
                    sem.semantic_graph.slice(0, 5).forEach(rel => {
                        semanticFeatures += `<div class="mt-1">${rel.source} → <span class="relation-tag">${rel.relation}</span> → ${rel.target}</div>`;
                    });
                    if (sem.semantic_graph.length > 5) {
                        semanticFeatures += `<div class="mt-1">...и еще ${sem.semantic_graph.length - 5} связей</div>`;
                    }
                    semanticFeatures += `</div>`;
                }
            }

            let similarHtml = '';
            if (result.similar_articles && result.similar_articles.length > 0) {
                similarHtml = `<div class="mt-3">
                    <h6>Похожие публикации:</h6>
                    <ul>`;
                result.similar_articles.forEach(article => {
                    similarHtml += `<li><a href="${article.link}" target="_blank">${article.title}</a></li>`;
                });
                similarHtml += `</ul></div>`;
            }

            const detailsHtml = `
                <h5>${result.TI || 'No title'}</h5>
                <p class="text-muted small">
                    ${result.DP || 'N/A'} | ${result.Source || 'N/A'} | ${result.LA || 'N/A'}
                </p>

                <div class="mb-3">
                    <span class="topic-badge">${result.Topic || 'No topic'}</span>
                </div>

                <div class="mb-3">
                    <strong>Авторы:</strong> ${authors}
                </div>

                <div class="mb-3">
                    <strong>Ключевые слова:</strong> 
                    <div>${keywords}</div>
                </div>

                ${topicTags}

                <div class="mb-3">
                    <strong>Аннотация:</strong>
                    <p>${result.AB || 'No abstract'}</p>
                </div>

                <div class="mb-3">
                    <strong>Сущности:</strong>
                    ${entitiesHtml || '<p>Не найдено</p>'}
                </div>

                ${semanticFeatures}

                ${similarHtml}
            `;

            document.getElementById('document-details').innerHTML = detailsHtml;

            // Прокрутка к деталям
            document.querySelector('.sticky-top').scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""

ABOUT_HTML = """
<!DOCTYPE html><html lang="ru"><head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>О проекте</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
            min-height: 100vh;
            padding: 2rem 0;
        }
        .about-header {
            background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
            color: white;
            padding: 4rem 0;
            border-radius: 0 0 30% 30%;
            margin-bottom: 3rem;
        }
        .feature-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            height: 100%;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
            transition: transform 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-10px);
        }
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1.5rem;
            color: #1a2980;
        }
        .team-member {
            text-align: center;
            margin-bottom: 2rem;
        }
        .team-member img {
            width: 150px;
            height: 150px;
            object-fit: cover;
            border-radius: 50%;
            border: 5px solid #e3f2fd;
            margin-bottom: 1rem;
        }
        .tech-logo {
            height: 60px;
            margin: 1rem;
            opacity: 0.7;
            transition: opacity 0.3s ease;
        }
        .tech-logo:hover {
            opacity: 1;
        }
        .btn-back {
            background: #1a2980;
            color: white;
            padding: 0.6rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .btn-back:hover {
            background: #0d1a5a;
            transform: translateY(-2px);
        }
    </style>
</head>
<body>
    <div class="about-header text-center">
        <div class="container">
            <h1 class="display-4 mb-3">О проекте</h1>
            <p class="lead">Использование искусственного интеллекта для семантического поиска в медицинских исследованиях</p>
        </div>
    </div>

    <div class="container">
        <div class="row justify-content-center mb-5">
            <div class="col-lg-8 text-center">
                <p class="lead mb-4">
                    Наш проект создан для помощи исследователям, врачам и студентам в поиске и анализе научных статей 
                    с использованием современных технологий искусственного интеллекта.
                </p>
                <p>
                    Система объединяет возможности семантического поиска, обработки естественного языка и визуализации данных, 
                    предоставляя мощный инструмент для научной работы.
                </p>
            </div>
        </div>

        <h2 class="text-center mb-4">Ключевые возможности</h2>
        <div class="row mb-5">
            <div class="col-md-4 mb-4">
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h4>Семантический поиск</h4>
                    <p>Поиск статей по смыслу, а не только по ключевым словам, с использованием нейросетевых моделей</p>
                </div>
            </div>
            <div class="col-md-4 mb-4">
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <h4>Анализ связей</h4>
                    <p>Выявление скрытых взаимосвязей между исследованиями, авторами и терминами</p>
                </div>
            </div>
            <div class="col-md-4 mb-4">
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h4>Визуализация данных</h4>
                    <p>Интерактивные графики и диаграммы для анализа тенденций и паттернов</p>
                </div>
            </div>
        </div>

        <h2 class="text-center mb-4">Технологии</h2>
        <div class="text-center mb-5">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Jupyter_logo.svg/1200px-Jupyter_logo.svg.png" class="tech-logo" alt="Python">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png" class="tech-logo" alt="Python">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/1200px-Tensorflow_logo.svg.png" class="tech-logo" alt="TensorFlow">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/1200px-TensorFlowLogo.svg.png" class="tech-logo" alt="TensorFlow">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/Figma-logo.svg/1200px-Figma-logo.svg.png" class="tech-logo" alt="Figma">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Node.js_logo.svg/1200px-Node.js_logo.svg.png" class="tech-logo" alt="Node.js">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/1200px-React-icon.svg.png" class="tech-logo" alt="React">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Amazon_Web_Services_Logo.svg/1200px-Amazon_Web_Services_Logo.svg.png" class="tech-logo" alt="AWS">
        </div>

        <h2 class="text-center mb-4">Наша команда</h2>
        <div class="row justify-content-center mb-5">
            <div class="col-md-3">
                <div class="team-member">
                    <div style="background: #e3f2fd; width: 150px; height: 150px; border-radius: 50%; margin: 0 auto 1rem;"></div>
                    <h5>Иван Петров</h5>
                    <p>Руководитель проекта</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="team-member">
                    <div style="background: #e3f2fd; width: 150px; height: 150px; border-radius: 50%; margin: 0 auto 1rem;"></div>
                    <h5>Мария Сидорова</h5>
                    <p>Data Scientist</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="team-member">
                    <div style="background: #e3f2fd; width: 150px; height: 150px; border-radius: 50%; margin: 0 auto 1rem;"></div>
                    <h5>Алексей Иванов</h5>
                    <p>Разработчик</p>
                </div>
            </div>
            <div class="col-md-3">
                <div class="team-member">
                    <div style="background: #e3f2fd; width: 150px; height: 150px; border-radius: 50%; margin: 0 auto 1rem;"></div>
                    <h5>Екатерина Смирнова</h5>
                    <p>Медицинский эксперт</p>
                </div>
            </div>
        </div>

        <div class="text-center mb-5">
            <h4 class="mb-4">Связаться с нами</h4>
            <p>Email: research@rosnou.ru</p>
            <p>Телефон: +7 (495) 123-45-67</p>
            <p>Адрес: Москва, ул. Радио, д. 22, офис 501</p>
        </div>

        <div class="text-center">
            <a href="/" class="btn-back">← На главную</a>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""


# Основные функции обработки
def detect_language(text):
    if not text or not isinstance(text, str) or not text.strip():
        return 'en'
    try:
        return detect(text)
    except:
        return 'en'


def preprocess_text(text, lang='en', additional_stop_words=None):
    if not text or not isinstance(text, str) or not text.strip():
        return []

    nlp = nlp_models.get(lang, nlp_models.get('en'))
    if not nlp:
        return []

    try:
        stop_words = set(stopwords.words(lang if lang in ['en', 'ru'] else 'en'))
        if additional_stop_words:
            stop_words.update(additional_stop_words)

        tokens = []
        chunk_size = 500000
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            doc = nlp(chunk.lower())
            tokens.extend([
                token.lemma_
                for token in doc
                if token.is_alpha and token.text not in stop_words and len(token.text) > 2
            ])

        return tokens
    except Exception as e:
        logger.error(f"Error in preprocess_text: {str(e)}")
        return []


def extract_medical_terms(text, lang='en'):
    if not text or not isinstance(text, str) or not text.strip():
        return []

    nlp = nlp_models.get(lang, nlp_models.get('en'))
    if not nlp:
        return []

    try:
        doc = nlp(text)
        matcher = Matcher(nlp.vocab)
        patterns = MEDICAL_PATTERNS.get(lang, MEDICAL_PATTERNS['en'])

        for i, pattern in enumerate(patterns):
            matcher.add(f"MED_TERM_{i}", [pattern])

        matches = matcher(doc)
        terms = []

        for match_id, start, end in matches:
            span = doc[start:end]
            term_text = span.text
            if 3 <= len(term_text) <= 50 and term_text not in terms:
                terms.append(term_text)

        # Добавление именованных сущностей
        for ent in doc.ents:
            if any(ent.label_ in labels for labels in MEDICAL_ENTITIES.values()):
                if ent.text not in terms:
                    terms.append(ent.text)

        return terms[:20]
    except Exception as e:
        logger.error(f"Error extracting terms: {str(e)}")
        return []


def get_umls_info(term):
    """Получение информации о термине из UMLS"""
    if term in term_cache:
        return term_cache[term]

    term_info = {
        'cui': None,
        'semantic_types': [],
        'is_disease': False
    }

    try:
        response = requests.get(
            "https://uts-ws.nlm.nih.gov/rest/search/current",
            params={"string": term, "apiKey": UMLS_API_KEY, "sabs": ",".join(PRIO_SABS)},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if 'result' in data and data['result']['results']:
            cui = data['result']['results'][0]['ui']
            resp = requests.get(
                f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}",
                params={"apiKey": UMLS_API_KEY},
                timeout=10
            )
            resp.raise_for_status()
            d = resp.json()
            sem_types = [st['name'] for st in d['result']['semanticTypes']]
            term_info = {
                'cui': cui,
                'semantic_types': sem_types,
                'is_disease': 'Disease or Syndrome' in sem_types or 'Neoplastic Process' in sem_types
            }

    except Exception as e:
        logger.error(f"UMLS query error for {term}: {str(e)}")

    term_cache[term] = term_info
    # Сохраняем кэш после каждого запроса
    try:
        with open(TERM_CACHE_FILE, 'wb') as f:
            pickle.dump(term_cache, f)
    except Exception as e:
        logger.error(f"Error saving term cache: {str(e)}")

    return term_info


def get_semantic_features(text, lang='ru'):
    """Извлечение семантических особенностей с помощью Pullenti"""
    features = {
        'synonyms': defaultdict(list),
        'homophones': defaultdict(list),
        'antonyms': defaultdict(list),
        'homographs': defaultdict(list),
        'semantic_graph': []
    }

    if lang != 'ru' or not text.strip():
        return features

    try:
        # Обработка текста
        analyzer = pullenti.ProcessorService.create_processor()
        analysis = analyzer.process(text)

        # Извлечение ключевых терминов
        keywords = []
        for r in analysis.entities:
            if isinstance(r, (pullenti.ner.titlepage.TitlePageReferent,
                              pullenti.ner.definition.DefinitionReferent,
                              pullenti.ner.keyword.KeywordReferent)):
                kw_text = r.to_string()
                if kw_text not in keywords:
                    keywords.append(kw_text)

        # Построение семантического графа
        for i in range(len(analysis.entities)):
            for j in range(i + 1, len(analysis.entities)):
                ent1 = analysis.entities[i]
                ent2 = analysis.entities[j]

                # Проверка наличия связи между сущностями
                if ent1 in ent2.relations or ent2 in ent1.relations:
                    relation_type = "RELATED"
                    features['semantic_graph'].append({
                        'source': ent1.to_string(),
                        'target': ent2.to_string(),
                        'relation': relation_type
                    })

        # Извлечение синонимов и антонимов
        for ent in analysis.entities:
            ent_text = ent.to_string()

            # Поиск синонимов
            if hasattr(ent, "synonyms"):
                for syn in ent.synonyms:
                    if isinstance(syn, Referent):
                        features['synonyms'][ent_text].append(syn.to_string())

            # Поиск антонимов
            if hasattr(ent, "antonyms"):
                for ant in ent.antonyms:
                    if isinstance(ant, Referent):
                        features['antonyms'][ent_text].append(ant.to_string())

        return features

    except Exception as e:
        logger.error(f"Pullenti processing error: {str(e)}")
        return features


def extract_entities(text, lang='en'):
    entities = {
        'diseases': [],
        'drugs': [],
        'procedures': [],
        'symptoms': [],
        'anatomy': [],
        'genes': [],
        'proteins': [],
        'semantic_features': {}
    }

    try:
        nlp = nlp_models.get(lang, nlp_models.get('en'))
        if nlp:
            doc = nlp(text)

            # Извлечение именованных сущностей
            for ent in doc.ents:
                if ent.label_ in MEDICAL_ENTITIES['diseases'] and ent.text not in entities['diseases']:
                    entities['diseases'].append(ent.text)
                elif ent.label_ in MEDICAL_ENTITIES['drugs'] and ent.text not in entities['drugs']:
                    entities['drugs'].append(ent.text)
                elif ent.label_ in MEDICAL_ENTITIES['procedures'] and ent.text not in entities['procedures']:
                    entities['procedures'].append(ent.text)
                elif ent.label_ in MEDICAL_ENTITIES['symptoms'] and ent.text not in entities['symptoms']:
                    entities['symptoms'].append(ent.text)
                elif ent.label_ in MEDICAL_ENTITIES['anatomy'] and ent.text not in entities['anatomy']:
                    entities['anatomy'].append(ent.text)
                elif ent.label_ in MEDICAL_ENTITIES['genes'] and ent.text not in entities['genes']:
                    entities['genes'].append(ent.text)
                elif ent.label_ in MEDICAL_ENTITIES['proteins'] and ent.text not in entities['proteins']:
                    entities['proteins'].append(ent.text)

            # Дополнительное извлечение терминов
            medical_terms = extract_medical_terms(text, lang)
            for term in medical_terms:
                umls_info = get_umls_info(term)
                sem_types = umls_info.get('semantic_types', [])

                if 'Disease or Syndrome' in sem_types or 'Neoplastic Process' in sem_types:
                    if term not in entities['diseases']:
                        entities['diseases'].append(term)
                elif 'Pharmacologic Substance' in sem_types:
                    if term not in entities['drugs']:
                        entities['drugs'].append(term)
                elif 'Body Part, Organ, or Organ Component' in sem_types:
                    if term not in entities['anatomy']:
                        entities['anatomy'].append(term)
                elif 'Sign or Symptom' in sem_types:
                    if term not in entities['symptoms']:
                        entities['symptoms'].append(term)
                elif 'Therapeutic or Preventive Procedure' in sem_types:
                    if term not in entities['procedures']:
                        entities['procedures'].append(term)
                elif 'Gene or Genome' in sem_types:
                    if term not in entities['genes']:
                        entities['genes'].append(term)
                elif 'Amino Acid, Peptide, or Protein' in sem_types:
                    if term not in entities['proteins']:
                        entities['proteins'].append(term)

        # Добавляем семантические особенности для русского текста
        if lang == 'ru':
            semantic_features = get_semantic_features(text, lang)
            entities['semantic_features'] = semantic_features

        return {k: v for k, v in entities.items() if v}
    except Exception as e:
        logger.error(f"Error in extract_entities: {str(e)}")
        return entities


def detect_topic_tags(text, lang='en'):
    """Определение тематических тегов на основе медицинских тем"""
    tokens = preprocess_text(text, lang)
    if not tokens:
        return []

    topic_scores = defaultdict(int)
    for topic, keywords in MEDICAL_TOPICS.items():
        for keyword in keywords:
            if keyword in tokens:
                topic_scores[topic] += 1

    # Сортируем темы по количеству совпадений
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

    # Возвращаем до 3 наиболее релевантных тем
    return [topic for topic, score in sorted_topics[:3]]


def perform_topic_modeling(documents):
    """Расширенное тематическое моделирование с использованием нескольких алгоритмов"""
    if not documents or not any(d and d.strip() for d in documents if isinstance(d, str)):
        return ["Общая медицина"] * len(documents) if documents else []

    topics = []
    valid_docs = [d for d in documents if isinstance(d, str) and d.strip()]

    try:
        # Попытка 1: LDA (Latent Dirichlet Allocation)
        if len(valid_docs) >= 3:
            try:
                lang = detect_language(" ".join(valid_docs))
                stop_words = stopwords.words(lang if lang in ['en', 'ru'] else 'en')

                vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words)
                dtm = vectorizer.fit_transform(valid_docs)

                n_topics = min(5, len(valid_docs))
                lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
                lda.fit(dtm)

                feature_names = vectorizer.get_feature_names_out()
                for i in range(len(documents)):
                    if isinstance(documents[i], str) and documents[i].strip():
                        topic_vec = lda.transform(vectorizer.transform([documents[i]]))[0]
                        top_topic = topic_vec.argmax()
                        top_words_idx = lda.components_[top_topic].argsort()[::-1][:3]
                        top_words = [feature_names[idx] for idx in top_words_idx]
                        topics.append(", ".join(top_words))
                    else:
                        topics.append("Общая медицина")
            except Exception as e:
                logger.error(f"LDA error: {str(e)}")
                topics = []

        # Попытка 2: KeyBERT (легковесная альтернатива)
        if not topics and len(valid_docs) >= 2:
            try:
                from keybert import KeyBERT

                kw_model = KeyBERT()
                for doc in valid_docs:
                    keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words='english')
                    if keywords:
                        topics.append(", ".join([kw[0] for kw in keywords[:2]]))
                    else:
                        topics.append("Общая медицина")
            except ImportError:
                logger.warning("KeyBERT not installed, skipping")
            except Exception as e:
                logger.error(f"KeyBERT error: {str(e)}")
                topics = []

        # Резервный метод: ключевые слова
        if not topics:
            for doc in valid_docs:
                lang = detect_language(doc)
                tags = detect_topic_tags(doc, lang)

                if tags:
                    topics.append(", ".join(tags[:2]))
                elif 'kag_model' in MODEL_CACHE and 'kag_tokenizer' in MODEL_CACHE:
                    input_text = f"Generate a short topic name for this medical text: {doc[:500]}"
                    inputs = MODEL_CACHE['kag_tokenizer'](input_text, return_tensors="pt")
                    outputs = MODEL_CACHE['kag_model'].generate(**inputs, max_length=30)
                    topic = MODEL_CACHE['kag_tokenizer'].decode(outputs[0], skip_special_tokens=True)
                    topics.append(topic)
                else:
                    terms = extract_medical_terms(doc)[:3]
                    topics.append(f"{', '.join(terms)}" if terms else "Медицинские исследования")

            # Для невалидных документов
            for _ in range(len(documents) - len(valid_docs)):
                topics.append("Общая медицина")

    except Exception as e:
        logger.error(f"Topic modeling error: {str(e)}")
        topics = ["Медицинские исследования"] * len(documents)

    return topics


def create_cluster_visualization(documents, topics):
    if not documents or not topics or len(documents) < 2:
        return ""

    try:
        valid_docs = [d for d in documents if isinstance(d, str) and d.strip()]
        if len(valid_docs) < 2:
            return ""

        # Используем TF-IDF если SentenceTransformer недоступен
        if MODEL_CACHE.get('sentence_model') is None:
            logger.info("Using TF-IDF for clustering")
            vectorizer = TfidfVectorizer()
            embeddings = vectorizer.fit_transform(valid_docs).toarray()
        else:
            embeddings = MODEL_CACHE['sentence_model'].encode(valid_docs)

        # Уменьшение размерности
        n_components = min(3, len(valid_docs), embeddings.shape[0])
        if n_components < 2:
            return ""

        try:
            pca = PCA(n_components=n_components)
            coords = pca.fit_transform(embeddings)
        except:
            coords = embeddings[:, :n_components]

        if coords.shape[1] < 3:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 3 - coords.shape[1]))])

        # Создание 3D визуализации
        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers+text',
            marker=dict(
                size=8,
                color=list(range(len(topics))),
                colorscale='Viridis',
                showscale=True,
                opacity=0.8
            ),
            text=[textwrap.shorten(t, width=20, placeholder="...") for t in topics],
            textposition="top center",
            hovertemplate='<b>Тема:</b> %{text}<br><b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Z:</b> %{z}'
        )])

        fig.update_layout(
            title="Тематические кластеры",
            scene=dict(
                xaxis_title="Компонент 1",
                yaxis_title="Компонент 2",
                zaxis_title="Компонент 3"
            ),
            height=500,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Cluster visualization error: {str(e)}")
        return ""


def create_network_visualization(results):
    if len(results) < 3:
        return ""

    try:
        G = nx.Graph()

        # Добавление узлов (статьи)
        for i, res in enumerate(results):
            title = res.get('TI', f"Статья {i + 1}")
            G.add_node(i, label=title[:50], size=10, topic=res.get('Topic', 'Общая'))

        # Добавление связей на основе общих терминов
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                # Общие заболевания
                diseases_i = set(res.get('entities', {}).get('diseases', []))
                diseases_j = set(res.get('entities', {}).get('diseases', []))
                disease_overlap = len(diseases_i & diseases_j)

                # Общие препараты
                drugs_i = set(res.get('entities', {}).get('drugs', []))
                drugs_j = set(res.get('entities', {}).get('drugs', []))
                drugs_overlap = len(drugs_i & drugs_j)

                # Общая мера сходства
                similarity = disease_overlap * 0.6 + drugs_overlap * 0.4

                if similarity > 0.5:
                    G.add_edge(i, j, weight=similarity)

        if not G.edges:
            return ""

        # Позиционирование узлов
        pos = nx.spring_layout(G, seed=42)

        # Подготовка данных для Plotly
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        node_text = []
        node_color = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            node_color.append(hash(G.nodes[node]['topic']) % 10)

        # Создание визуализации
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Rainbow',
                color=node_color,
                size=15,
                line_width=1
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title='Семантическая сеть публикаций',
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=500
                        ))

        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Network visualization error: {str(e)}")
        return ""


def plot_topic_frequency(results, topics):
    if not results or not topics:
        return ""

    try:
        years = []
        for r in results:
            dp = r.get('DP', '')
            if dp and dp != 'N/A':
                try:
                    year = int(dp.split()[0]) if ' ' in dp else int(dp)
                    if 1900 <= year <= 2025:
                        years.append(year)
                except:
                    pass

        if not years or len(years) != len(topics):
            return ""

        year_topic = defaultdict(Counter)
        for y, t in zip(years, topics):
            if t and t != "Общая медицина":
                topic_simple = t.split(',')[0] if ',' in t else t[:20]
                year_topic[y][topic_simple] += 1

        if not year_topic:
            return ""

        sorted_years = sorted(year_topic.keys())
        all_topics = list(set(t for c in year_topic.values() for t in c))[:10]

        plt.figure(figsize=(12, 6))
        plt.clf()

        matrix = np.zeros((len(all_topics), len(sorted_years)))
        for i, y in enumerate(sorted_years):
            for j, t in enumerate(all_topics):
                matrix[j, i] = year_topic[y].get(t, 0)

        colors = plt.cm.tab20(np.linspace(0, 1, len(all_topics)))

        plt.stackplot(sorted_years, matrix, labels=all_topics, colors=colors, alpha=0.8)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
        plt.title("Распределение тем по годам", fontsize=14, fontweight='bold')
        plt.xlabel("Год", fontsize=12)
        plt.ylabel("Количество статей", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Year plot error: {str(e)}")
        return ""


def create_wordcloud(results):
    if not results:
        return ""
    try:
        text = ' '.join(' '.join(r.get('KW', [])) for r in results)
        if not text:
            return ""
        wc = WordCloud(
            background_color="white",
            max_words=100,
            colormap='viridis',
            width=800,
            height=400
        )
        wc.generate(text)
        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Wordcloud error: {str(e)}")
        return ""


def plot_topics_distribution(topics):
    if not topics:
        return ""

    try:
        # Фильтрация общих тем
        filtered_topics = [t for t in topics if t not in ["Общая медицина", "Медицинские исследования"]]

        if not filtered_topics:
            return ""

        topic_counts = Counter(filtered_topics)
        top_topics = dict(topic_counts.most_common(10))

        if not top_topics:
            return ""

        plt.figure(figsize=(10, 6))
        plt.barh(list(top_topics.keys()), list(top_topics.values()), color='#1a2980')
        plt.title('Распределение тематик', fontsize=14)
        plt.xlabel('Количество статей', fontsize=12)
        plt.ylabel('Тема', fontsize=12)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Topics distribution plot error: {str(e)}")
        return ""


def create_dendrogram_visualization(results):
    """Иерархическая кластеризация медицинских терминов"""
    try:
        # Сбор всех терминов
        all_terms = []
        for r in results:
            if 'entities' in r:
                for cat in ['diseases', 'drugs', 'procedures', 'symptoms']:
                    if cat in r['entities']:
                        all_terms.extend(r['entities'][cat])

        if not all_terms:
            return ""

        # Подсчет частоты терминов
        term_freq = Counter(all_terms)
        top_terms = [term for term, _ in term_freq.most_common(15)]

        # Создание матрицы встречаемости
        matrix = np.zeros((len(results), len(top_terms)))
        for i, r in enumerate(results):
            for j, term in enumerate(top_terms):
                for cat in ['diseases', 'drugs', 'procedures', 'symptoms']:
                    if cat in r.get('entities', {}) and term in r['entities'][cat]:
                        matrix[i, j] = 1

        # Иерархическая кластеризация
        linkage_matrix = linkage(matrix.T, 'ward')

        # Визуализация
        plt.figure(figsize=(15, 10))
        dendrogram(
            linkage_matrix,
            labels=top_terms,
            orientation='right',
            leaf_font_size=12
        )
        plt.title('Иерархическая кластеризация медицинских терминов', fontsize=16)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Dendrogram error: {str(e)}")
        return ""


def create_semantic_graph_visualization(results):
    """Визуализация семантического графа из Pullenti"""
    try:
        # Сбор всех семантических связей
        all_relations = []
        node_set = set()

        for r in results:
            if 'entities' in r and 'semantic_features' in r['entities']:
                for rel in r['entities']['semantic_features'].get('semantic_graph', []):
                    all_relations.append(rel)
                    node_set.add(rel['source'])
                    node_set.add(rel['target'])

        if len(all_relations) < 3:
            return ""

        # Создание графа
        G = nx.DiGraph()

        # Добавление узлов
        for node in node_set:
            G.add_node(node, size=10)

        # Добавление связей
        for rel in all_relations:
            G.add_edge(rel['source'], rel['target'], label=rel['relation'])

        # Позиционирование
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # Подготовка для Plotly
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x = []
        node_y = []
        node_text = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        # Создание трассировок
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Rainbow',
                size=20,
                line_width=2
            )
        )

        # Создание фигуры
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Семантическая сеть терминов',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )

        # Добавление аннотаций для связей
        annotations = []
        for rel in all_relations[:20]:  # Ограничиваем количество аннотаций
            try:
                x0, y0 = pos[rel['source']]
                x1, y1 = pos[rel['target']]
                annotations.append(dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    xref="x", yref="y",
                    text=rel['relation'],
                    showarrow=False,
                    font=dict(size=10)
                ))
            except:
                continue

        fig.update_layout(annotations=annotations)
        return fig.to_html(full_html=False)

    except Exception as e:
        logger.error(f"Semantic graph error: {str(e)}")
        return ""


def create_timeline_visualization(results):
    """Временная шкала публикаций по темам"""
    try:
        # Сбор данных
        timeline_data = []
        for r in results:
            if 'DP' in r and r['DP'] != 'N/A' and 'Topic' in r:
                try:
                    year = int(r['DP'].split()[0])
                    timeline_data.append({
                        'year': year,
                        'topic': r['Topic'],
                        'title': r['TI'][:50] + ('...' if len(r['TI']) > 50 else ''),
                        'source': r['Source']
                    })
                except:
                    continue

        if not timeline_data:
            return ""

        # Создание DataFrame
        df = pd.DataFrame(timeline_data)

        # Группировка по годам и темам
        df_counts = df.groupby(['year', 'topic']).size().reset_index(name='count')

        # Визуализация
        plt.figure(figsize=(15, 8))
        sns.lineplot(
            data=df_counts,
            x='year',
            y='count',
            hue='topic',
            marker='o',
            linewidth=2.5
        )
        plt.title('Динамика публикаций по темам', fontsize=16)
        plt.xlabel('Год', fontsize=14)
        plt.ylabel('Количество статей', fontsize=14)
        plt.legend(title='Тема', loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Timeline error: {str(e)}")
        return ""


def create_correlation_matrix(results):
    """Матрица корреляции медицинских терминов"""
    try:
        # Сбор всех терминов
        all_terms = set()
        for r in results:
            if 'entities' in r:
                for cat in ['diseases', 'drugs', 'procedures', 'symptoms']:
                    if cat in r['entities']:
                        all_terms.update(r['entities'][cat])

        if len(all_terms) < 2:
            return ""

        # Выбор топ-15 терминов
        term_counts = Counter([term for r in results for cat in ['diseases', 'drugs', 'procedures', 'symptoms']
                               if 'entities' in r and cat in r['entities'] for term in r['entities'][cat]])
        top_terms = [term for term, _ in term_counts.most_common(15)]

        # Создание матрицы совместной встречаемости
        matrix = np.zeros((len(top_terms), len(top_terms)))

        for i, term1 in enumerate(top_terms):
            for j, term2 in enumerate(top_terms):
                if i != j:
                    count = 0
                    for r in results:
                        has_term1 = any(term1 in r['entities'].get(cat, []) for cat in
                                        ['diseases', 'drugs', 'procedures', 'symptoms'])
                        has_term2 = any(term2 in r['entities'].get(cat, []) for cat in
                                        ['diseases', 'drugs', 'procedures', 'symptoms'])
                        if has_term1 and has_term2:
                            count += 1
                    matrix[i, j] = count

        # Визуализация
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".0f",
            cmap="YlGnBu",
            xticklabels=top_terms,
            yticklabels=top_terms,
            linewidths=0.5
        )
        plt.title('Матрица совместной встречаемости терминов', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Correlation matrix error: {str(e)}")
        return ""


def create_venn_diagram(results):
    """Создание диаграммы Венна для топ-3 тем"""
    try:
        topics = [r.get('Topic', '') for r in results]
        if len(set(topics)) < 2:
            return ""

        # Выбор топ-3 тем
        topic_counts = Counter(topics)
        top_topics = [topic for topic, _ in topic_counts.most_common(3)]

        # Создание множеств документов по темам
        sets = []
        for topic in top_topics:
            indices = [i for i, t in enumerate(topics) if t == topic]
            sets.append(set(indices))

        # Проверка размеров множеств
        if len(sets) == 2:
            plt.figure(figsize=(8, 6))
            venn2(sets, set_labels=top_topics)
        elif len(sets) == 3:
            plt.figure(figsize=(10, 8))
            venn3(sets, set_labels=top_topics)
        else:
            return ""

        plt.title("Пересечение тематик", fontsize=14)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Venn diagram error: {str(e)}")
        return ""


def plot_topic_trends(results):
    """Линейный график изменения ключевых тем по годам"""
    try:
        # Сбор данных
        data = []
        for r in results:
            if 'DP' in r and r['DP'] != 'N/A' and 'Topic' in r:
                try:
                    year = int(r['DP'].split()[0])
                    topic = r['Topic']
                    if topic != "Общая медицина":
                        data.append({'year': year, 'topic': topic})
                except:
                    continue

        if not data:
            return ""

        df = pd.DataFrame(data)

        # Выбор топ-5 тем
        top_topics = df['topic'].value_counts().head(5).index.tolist()
        df = df[df['topic'].isin(top_topics)]

        # Группировка по годам и темам
        df_counts = df.groupby(['year', 'topic']).size().reset_index(name='count')

        # Визуализация
        plt.figure(figsize=(12, 6))
        for topic in top_topics:
            topic_data = df_counts[df_counts['topic'] == topic]
            plt.plot(topic_data['year'], topic_data['count'], marker='o', label=topic[:20])

        plt.title("Динамика ключевых тем по годам", fontsize=14)
        plt.xlabel("Год", fontsize=12)
        plt.ylabel("Количество публикаций", fontsize=12)
        plt.legend(title='Тематика', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Topic trends error: {str(e)}")
        return ""


def plot_term_frequency_by_year(results):
    """Гистограмма частотности терминов по годам"""
    try:
        # Сбор данных
        term_year_data = defaultdict(lambda: defaultdict(int))
        years = set()

        for r in results:
            if 'DP' in r and r['DP'] != 'N/A':
                try:
                    year = int(r['DP'].split()[0])
                    years.add(year)

                    # Сбор терминов из всех категорий
                    terms = []
                    if 'entities' in r:
                        for cat in ['diseases', 'drugs', 'procedures', 'symptoms']:
                            if cat in r['entities']:
                                terms.extend(r['entities'][cat])

                    # Подсчет терминов
                    for term in set(terms):
                        term_year_data[term][year] += 1
                except:
                    continue

        if not term_year_data or not years:
            return ""

        # Выбор топ-10 терминов
        term_counts = {term: sum(counts.values()) for term, counts in term_year_data.items()}
        top_terms = [term for term, _ in Counter(term_counts).most_common(10)]
        sorted_years = sorted(years)

        # Подготовка данных для гистограммы
        data = []
        for term in top_terms:
            counts = [term_year_data[term].get(year, 0) for year in sorted_years]
            data.append(counts)

        # Визуализация
        plt.figure(figsize=(12, 6))
        bar_width = 0.08
        index = np.arange(len(sorted_years))

        for i, term in enumerate(top_terms):
            plt.bar(index + i * bar_width, data[i], bar_width, label=term[:15])

        plt.title("Частотность терминов по годам", fontsize=14)
        plt.xlabel("Год", fontsize=12)
        plt.ylabel("Количество упоминаний", fontsize=12)
        plt.xticks(index + bar_width * 4, sorted_years)
        plt.legend(title='Термины', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Term frequency error: {str(e)}")
        return ""


def create_dashboard(results):
    """Создание комплексного дашборда с аналитикой"""
    dashboard = {}

    # 1. Иерархическая кластеризация терминов
    dashboard['dendrogram'] = create_dendrogram_visualization(results)

    # 2. Граф семантических связей
    dashboard['semantic_graph'] = create_semantic_graph_visualization(results)

    # 3. Временная шкала публикаций
    dashboard['timeline'] = create_timeline_visualization(results)

    # 4. Матрица корреляции терминов
    dashboard['correlation_matrix'] = create_correlation_matrix(results)

    return dashboard


def search_pubmed(query, year_start, year_end, selected_topics=None):
    results = []
    try:
        # Формирование запроса с учетом выбранных тем
        search_query = f"{query} AND ({year_start}:{year_end}[DP])"
        if selected_topics:
            topics_query = " OR ".join(selected_topics)
            search_query = f"({search_query}) AND ({topics_query})"

        handle = Entrez.esearch(
            db="pubmed",
            term=search_query,
            retmax=20
        )
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]

        if pmids:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
            records = list(Medline.parse(StringIO(handle.read())))
            handle.close()

            for i, r in enumerate(records):
                year_str = r.get('DP', '')
                if year_str:
                    try:
                        year = int(year_str.split()[0])
                        if year_start <= year <= year_end:
                            abstract = r.get('AB', '')
                            lang = detect_language(abstract if abstract else r.get('TI', ''))

                            # Извлечение ключевых слов
                            keywords = r.get('MH', [])
                            if not keywords and abstract:
                                keywords = extract_medical_terms(abstract, lang)[:8]

                            # Извлечение сущностей
                            entities = extract_entities(f"{r.get('TI', '')} {abstract}", lang)

                            result = {
                                'TI': r.get('TI', 'No title'),
                                'DP': year_str,
                                'AB': abstract if abstract else 'No abstract available',
                                'AU': r.get('AU', []),
                                'KW': keywords if keywords else [],
                                'LA': lang,
                                'Source': 'PubMed',
                                'PMID': pmids[i],
                                'entities': entities,
                                'diseases': entities.get('diseases', [])
                            }
                            results.append(result)
                    except Exception as e:
                        logger.error(f"PubMed processing error: {str(e)}")
    except Exception as e:
        logger.error(f"PubMed search error: {str(e)}")

    return results


def search_core(query, year_start, year_end, selected_topics=None):
    results = []
    try:
        # Формирование запроса с учетом выбранных тем
        search_query = f"{query} AND year: [{year_start} TO {year_end}]"
        if selected_topics:
            topics_query = " OR ".join(selected_topics)
            search_query = f"({search_query}) AND ({topics_query})"

        headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
        params = {
            "q": search_query,
            "limit": 20
        }
        response = requests.get(
            "https://api.core.ac.uk/v3/search/works",
            headers=headers,
            params=params,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get('results', []):
            abstract = item.get('abstract', '')
            lang = detect_language(abstract if abstract else item.get('title', ''))

            # Извлечение ключевых слов
            keywords = item.get('keywords', [])
            if not keywords and abstract:
                keywords = extract_medical_terms(abstract, lang)[:8]

            # Извлечение сущностей
            entities = extract_entities(f"{item.get('title', '')} {abstract}", lang)

            result = {
                'TI': item.get('title', 'No title'),
                'DP': str(item.get('yearPublished', 'N/A')),
                'AB': abstract if abstract else 'No abstract available',
                'AU': [author['name'] for author in item.get('authors', [])],
                'KW': keywords if keywords else [],
                'LA': lang,
                'Source': 'CORE',
                'ID': item.get('id'),
                'entities': entities,
                'diseases': entities.get('diseases', [])
            }
            results.append(result)
    except Exception as e:
        logger.error(f"CORE search error: {str(e)}")

    return results


def get_similar_articles(result):
    similar = []
    try:
        if result['Source'] == 'PubMed' and 'PMID' in result:
            handle = Entrez.elink(db="pubmed", id=result['PMID'], cmd="neighbor", retmax=3)
            record = Entrez.read(handle)
            handle.close()

            if record and record[0].get('LinkSetDb'):
                similar_pmids = [link['Id'] for link in record[0]['LinkSetDb'][0]['Link']]
                if similar_pmids:
                    handle = Entrez.efetch(db="pubmed", id=",".join(similar_pmids), rettype="medline", retmode="text")
                    records = list(Medline.parse(StringIO(handle.read())))
                    handle.close()
                    similar = [{
                        'title': r.get('TI', 'No title'),
                        'link': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                    } for pmid, r in zip(similar_pmids, records)]
        elif result['Source'] == 'CORE' and 'ID' in result:
            headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
            response = requests.get(
                f"https://api.core.ac.uk/v3/works/{result['ID']}/similar",
                headers=headers,
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                similar = [{
                    'title': item.get('title', 'No title'),
                    'link': item.get('downloadUrl', f"https://core.ac.uk/works/{item.get('id')}")
                } for item in data.get('results', [])[:3]]
    except Exception as e:
        logger.error(f"Similar articles error: {str(e)}")

    return similar


def generate_insights(results):
    if not results or not OPENAI_API_KEY:
        return ""

    try:
        import openai
        openai.api_key = OPENAI_API_KEY

        # Создаем сводку результатов
        summary = f"Поисковый запрос: {session.get('last_query', '')}\n\n"
        summary += f"Найдено статей: {len(results)}\n\n"

        # Анализ тем
        topics = [r.get('Topic', '') for r in results if r.get('Topic')]
        if topics:
            topic_counts = Counter(topics)
            top_topics = topic_counts.most_common(5)
            summary += "Основные темы:\n"
            for topic, count in top_topics:
                summary += f"- {topic}: {count} статей\n"
            summary += "\n"

        # Анализ заболеваний
        all_diseases = []
        for r in results:
            all_diseases.extend(r.get('entities', {}).get('diseases', []))

        if all_diseases:
            disease_counts = Counter(all_diseases)
            top_diseases = disease_counts.most_common(5)
            summary += "Наиболее изучаемые заболевания:\n"
            for disease, count in top_diseases:
                summary += f"- {disease}: {count} упоминаний\n"
            summary += "\n"

        # Генерация инсайтов с помощью GPT
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты - научный аналитик. Проанализируй данные и выдели ключевые инсайты."},
                {"role": "user",
                 "content": f"Вот сводка результатов поиска:\n\n{summary}\n\nВыдели 2-3 ключевых инсайта."}
            ],
            max_tokens=300,
            temperature=0.3
        )

        return response.choices[0].message['content'].strip()
    except ImportError:
        logger.warning("OpenAI library not installed, insights generation skipped")
        return ""
    except Exception as e:
        logger.error(f"Insights generation error: {str(e)}")
        return ""


def process_uploaded_file(file):
    if not file:
        return ""

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    text = ""

    try:
        # PDF файлы (работа в памяти)
        if ext == '.pdf':
            file_bytes = file.read()
            pdf_file = BytesIO(file_bytes)
            reader = PyPDF2.PdfReader(pdf_file)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        # DOCX файлы (работа с временным файлом)
        elif ext == '.docx':
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
                file.save(tmp.name)
            text = docx2txt.process(tmp.name)
            os.unlink(tmp.name)

        # Изображения (работа в памяти)
        elif ext in ('.png', '.jpg', '.jpeg', '.tiff'):
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                reader = easyocr.Reader(['en', 'ru'])
                result = reader.readtext(gray)
                text = " ".join([res[1] for res in result])

        # Текстовые файлы (работа в памяти)
        elif ext in ('.txt', '.csv'):
            text = file.read().decode('utf-8', errors='replace')

        # Excel файлы (работа в памяти)
        elif ext == '.xlsx':
            file_bytes = file.read()
            wb = openpyxl.load_workbook(BytesIO(file_bytes))
            sheet = wb.active
            text = "\n".join(" ".join(str(cell.value) for cell in row) for row in sheet.rows)
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        return ""

    return text


# Маршруты Flask
@app.route('/')
def home():
    medical_topics = list(MEDICAL_TOPICS.keys())
    return render_template_string(INDEX_HTML, medical_topics=medical_topics)


@app.route('/about')
def about():
    return render_template_string(ABOUT_HTML)


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    db = request.form.get('db', 'pubmed')
    year_start = request.form.get('year_start', '2015')
    year_end = request.form.get('year_end', '2025')
    exclude_terms = request.form.get('exclude_terms', '')
    selected_topics = request.form.get('selected_topics', '')

    # Сохраняем запрос в сессии
    session['last_query'] = query

    # Обработка выбранных тем
    selected_topics_list = []
    if selected_topics:
        selected_topics_list = [topic.strip() for topic in selected_topics.split(',')]

    # Проверка и преобразование параметров
    try:
        year_start = int(year_start)
        year_end = int(year_end)
        if year_start > year_end or year_start < 1900 or year_end > datetime.now().year:
            year_start, year_end = 2018, 2025
    except:
        year_start, year_end = 2018, 2025

    # Выполнение поиска
    results = []

    if db in ['pubmed', 'combined']:
        results.extend(search_pubmed(query, year_start, year_end, selected_topics_list))

    if db in ['core', 'combined']:
        results.extend(search_core(query, year_start, year_end, selected_topics_list))

    # Если результатов нет
    if not results:
        results = [{
            'TI': 'Результаты не найдены',
            'DP': 'N/A',
            'AB': 'Попробуйте изменить параметры поиска',
            'AU': [],
            'KW': [],
            'LA': 'en',
            'Source': db,
            'Topic': 'N/A',
            'entities': {},
            'diseases': []
        }]

    # Обработка результатов
    documents = [f"{r.get('TI', '')} {r.get('AB', '')}" for r in results]
    topics = perform_topic_modeling(documents)

    # Добавление тем, тегов и похожих статей
    for i, r in enumerate(results):
        if i < len(topics):
            r['Topic'] = topics[i]
        else:
            r['Topic'] = "Медицинские исследования"

        # Добавление тематических тегов
        lang = r.get('LA', 'en')
        text = f"{r.get('TI', '')} {r.get('AB', '')}"
        r['topic_tags'] = detect_topic_tags(text, lang)

        r['similar_articles'] = get_similar_articles(r)

    # Создание визуализаций
    visualizations = {
        'cluster_plot': create_cluster_visualization(documents, topics),
        'year_plot': plot_topic_frequency(results, topics),
        'wordcloud': create_wordcloud(results),
        'network_plot': create_network_visualization(results),
        'topics_distribution': plot_topics_distribution(topics),
        'venn_diagram': create_venn_diagram(results),
        'topic_trends': plot_topic_trends(results),
        'term_frequency': plot_term_frequency_by_year(results)
    }

    # Генерация аналитических инсайтов
    insights = generate_insights(results)

    # Создание дашборда
    dashboard = create_dashboard(results)

    return render_template_string(
        RESULTS_HTML,
        results=results,
        db=db,
        year_start=year_start,
        year_end=year_end,
        insights=insights,
        selected_topics=", ".join(selected_topics_list) if selected_topics_list else None,
        **visualizations,
        **dashboard
    )


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect('/')

    file = request.files['file']
    if file.filename == '':
        return redirect('/')

    text = process_uploaded_file(file)
    if not text:
        return render_template_string(
            RESULTS_HTML,
            results=[{
                'TI': 'Ошибка обработки',
                'DP': 'N/A',
                'AB': 'Не удалось извлечь текст из файла',
                'AU': [],
                'KW': [],
                'LA': 'en',
                'Source': 'Upload',
                'Topic': 'N/A',
                'entities': {},
                'diseases': []
            }],
            db='upload',
            year_start=2018,
            year_end=2025
        )

    lang = detect_language(text)
    entities = extract_entities(text, lang)
    keywords = extract_medical_terms(text, lang)[:10]
    topics = perform_topic_modeling([text])
    topic_tags = detect_topic_tags(text, lang)

    result = {
        'TI': f"Загруженный документ: {file.filename}",
        'DP': 'N/A',
        'AB': text[:2000] + ('...' if len(text) > 2000 else ''),
        'AU': [],
        'KW': keywords,
        'LA': lang,
        'Source': 'Upload',
        'Topic': topics[0] if topics else 'Анализ документа',
        'topic_tags': topic_tags,
        'entities': entities,
        'diseases': entities.get('diseases', [])
    }

    return render_template_string(
        RESULTS_HTML,
        results=[result],
        db='upload',
        year_start=2018,
        year_end=2025
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)