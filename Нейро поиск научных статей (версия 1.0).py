from flask import Flask, request, render_template_string
from Bio import Entrez
from Bio import Medline
from io import StringIO, BytesIO
import nltk
from nltk.corpus import stopwords, wordnet, cmudict
import spacy
from spacy.matcher import Matcher
from langdetect import detect
import PyPDF2
import docx2txt
import openpyxl
import requests
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from collections import Counter, defaultdict
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle
import os
from bertopic import BERTopic
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import warnings
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import networkx as nx
import cv2  # Added for contour detection and rotation bounding box
import easyocr  # Added for advanced OCR with rotation support
from pullenti.ner.ProcessorService import ProcessorService  # Pullenti for entity extraction
from pullenti.ner.SourceOfAnalysis import SourceOfAnalysis
import numpy as np
import time  # For ABBYY polling
from wordcloud import WordCloud  # For improved graphics: word cloud

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('cmudict', quiet=True)

app = Flask(__name__)
Entrez.email = "yurchak777x@gmail.com"

# Initialize Pullenti
ProcessorService.initialize()
pullenti_processor = ProcessorService.create_processor()

# Initialize spaCy models
spacy_models = {'en': 'en_core_web_sm', 'ru': 'ru_core_news_sm', 'es': 'es_core_news_sm', 'fr': 'fr_core_news_sm',
                'de': 'de_core_news_sm'}
nlp_models = {}
for lang, model in spacy_models.items():
    try:
        nlp_models[lang] = spacy.load(model)
    except:
        logger.warning(f"Could not load spaCy model for {lang}")

# Initialize transformer models
try:
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
    sentence_model = SentenceTransformer(
        'all-MiniLM-L6-v2')  # Kept as is; consider 'paraphrase-multilingual-mpnet-base-v2' if available
except Exception as e:
    logger.error(f"Error loading transformer models: {e}")
    t5_tokenizer = None
    t5_model = None
    sentence_model = None

# Cache and configurations
TERM_CACHE_FILE = "term_cache.pkl"
term_cache = {}
if os.path.exists(TERM_CACHE_FILE):
    try:
        with open(TERM_CACHE_FILE, 'rb') as f:
            term_cache = pickle.load(f)
    except:
        term_cache = {}

CORE_API_KEY = os.getenv("CORE_API_KEY", "3qj6Ggun9KY1AJPOlvLRHpmVUxkZEeCo")
UMLS_API_KEY = os.getenv("UMLS_API_KEY", "5266ecdf-b5ef-4f6c-922b-55c224f950b9")
NANONETS_API_KEY = os.getenv("dk_bIGNaCGH-qw8PvsZu9coMcu-CGXC7")  # New: For Nanonets API
NANONETS_MODEL_ID = os.getenv("1927b9d9-efd3-4345-8911-fa8203123d11")  # Assume a pre-trained custom model ID
ABBYY_APP_ID = os.getenv("51a7f7a1-ccba-412a-b831-59a630780da0")  # New: For ABBYY
ABBYY_PASSWORD = os.getenv("GETE#911*Uc13")  # New
prio_sabs = ["SNOMEDCT_US", "MSH", "MEDCIN", "ICD10CM", "RXNORM", "CPT", "LOINC"]

# Exclude terms
EXCLUDE_TERMS = {'5-hour', '4-week', 'people', 'doctors', 'patients', 'numbers', 'children', 'pre', 'OBJECTIVE',
                 'purpose', 'background', 'conclusions', 'methods', 'results', 'PURPOSE', 'BACKGROUND', 'CONCLUSIONS',
                 'METHODS', 'RESULTS', 'objective', 'kg', 'g', 'cm', 'mm'}

# Rare terms log file
RARE_TERMS_FILE = "rare_terms.txt"

# Enhanced HTML templates with added exclude terms input and wordcloud
INDEX_HTML = """
<!DOCTYPE html><html><head><title>Нейро поиск исследовательских статей</title>
<style>
    /* ... existing styles ... */
    .logo { max-width: 200px; display: block; margin: 0 auto 20px; }
    body { font-family: 'Segoe UI', Arial; margin: 20px; background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
    h1 { color: #333; text-align: center; margin-bottom: 30px; }
    form { margin-bottom: 20px; }
    input, select { padding: 10px; margin: 5px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; }
    input[type="submit"] { background: #007bff; color: white; cursor: pointer; border: none; padding: 12px 30px; }
    input[type="submit"]:hover { background: #0056b3; }
    .form-row { display: flex; justify-content: center; flex-wrap: wrap; margin: 10px 0; }
</style></head>
<body>
<div class="container">
    <img src="https://upload.wikimedia.org/wikipedia/ru/8/86/Rosnou_logo.png" alt="РосНОУ логотип" class="logo">
    <h1>Нейро поиск исследовательских статей</h1>
    <a href="/about" style="display: block; text-align: center; margin-top: 20px; color: #007bff;">О проекте</a>
    <form action="/search" method="post">
        <div class="form-row">
            <input type="text" name="query" placeholder="Введите поисковый запрос" required style="width: 400px;">
            <select name="db">
                <option value="pubmed">PubMed</option>
                <option value="core">CORE</option>
            </select>
        </div>
        <div class="form-row">
            <input type="number" name="year_start" placeholder="Год начала поиска статей" min="1900" max="2025" value="2015">
            <input type="number" name="year_end" placeholder="Год окончания поиска статей" min="1900" max="2025" value="2025">
            <input type="text" name="exclude_terms" placeholder="Исключаемые термины (через запятую)">
            <input type="submit" value="Search">
        </div>
    </form>
    <h2>Загрузить документ</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <div class="form-row">
            <input type="file" name="file" accept=".pdf,.docx,.xlsx,.png,.jpg,.tiff">
            <input type="submit" value="Upload">
        </div>
    </form>
</div>
</body></html>
"""

RESULTS_HTML = """
<!DOCTYPE html><html><head><title>Результаты поиска статей</title>
<style>
    body { font-family: 'Segoe UI', Arial; margin: 20px; background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); }
    .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
    h1, h2 { color: #333; }
    table { border-collapse: collapse; width: 100%; background: #fff; margin: 20px 0; }
    th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
    th { background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); color: white; }
    .entity { background: #d4edda; padding: 2px 4px; border-radius: 3px; }
    .topic-badge { background: #007bff; color: white; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
    .keyword { background: #ffc107; color: #333; padding: 2px 6px; border-radius: 3px; margin: 2px; display: inline-block; }
    a { color: #007bff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .error-message { color: #dc3545; font-weight: bold; }
    .viz-container { margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px; }
    .viz-title { font-size: 18px; font-weight: bold; margin-bottom: 15px; color: #495057; }
    .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 30px; grid-auto-rows: minmax(500px, auto); } /* Improved for even, sequential display */
    .back-button { display: inline-block; margin-top: 20px; padding: 10px 20px; background: #6c757d; color: white; border-radius: 5px; }
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body>
<div class="container">
    <h1> Результаты поиска для {{ db|upper }} ({{ year_start }}-{{ year_end }})</h1>
    {% if not results or results[0].get('TI') == 'Результаты не найдены' %}
        <p class="error-message">Результаты не найдены.</p>
    {% else %}
        <table>
            <tr>
                <th>Title</th>
                <th>Language</th>
                <th>Year</th>
                <th>Authors</th>
                <th>Keywords</th>
                <th>Abstract</th>
                <th>Main Topic</th>
                <th>Entities</th>
                <th>Diseases</th>
                <th>Similar Articles</th>
            </tr>
            {% for result in results %}
            <tr>
                <td>{{ result.get('TI', 'No title') }}</td>
                <td>{{ result.get('LA', 'Unknown') }}</td>
                <td>{{ result.get('DP', 'No date') }}</td>
                <td>{{ result.get('AU', ['N/A'])|join(', ') if result.get('AU') else 'N/A' }}</td>
                <td>
                    {% for kw in result.get('KW', []) %}
                        <span class="keyword">{{ kw }}</span>
                    {% endfor %}
                    {% if not result.get('KW') %}
                        <span style="color: #999;">No keywords</span>
                    {% endif %}
                </td>
                <td>{{ result.get('AB', 'No abstract')|safe }}</td>
                <td><span class="topic-badge">{{ result.get('Topic', 'No topic') }}</span></td>
                <td>
                    <ul>
                    {% for type, vals in result.get('entities', {}).items() %}
                        <li>{{ type.capitalize() }}: {{ vals|join(', ') if vals else 'None' }}</li>
                    {% endfor %}
                    </ul>
                </td>
                <td>{{ result.get('diseases', [])|join(', ') if result.get('diseases') else 'None' }}</td>
                <td>
                    {% for article in result.get('similar_articles', []) %}
                        <a href="{{ article.link }}" target="_blank">{{ article.title }}</a><br>
                    {% endfor %}
                </td>
            </tr>
            {% endfor %}
        </table>

        <div class="viz-container">
            <h2>Анализ & Визуализация</h2>

            <div class="chart-grid">
                {% if cluster_plot %}
                <div>
                    <div class="viz-title">Тематические кластеры</div>
                    <div id="cluster_plot"></div>
                    {{ cluster_plot|safe }}
                </div>
                {% endif %}

                {% if year_plot %}
                <div>
                    <div class="viz-title">Распределение тем по годам</div>
                    <img src="data:image/png;base64,{{ year_plot }}" alt="Частота распределения тем по статьям">
                </div>
                {% endif %}

                {% if terms_timeline %}
                <div>
                    <div class="viz-title">Эволюция уникальных терминов</div>
                    <div id="terms_timeline"></div>
                    {{ terms_timeline|safe }}
                </div>
                {% endif %}

                {% if venn_plot %}
                <div>
                    <div class="viz-title">Перекрытие тем (диаграмма Венна)</div>
                    <img src="data:image/png;base64,{{ venn_plot }}" alt="Topic Venn">
                </div>
                {% endif %}

                {% if disease_graph %}
                <div>
                    <div class="viz-title">Граф распределения статей по заболеваниям</div>
                    <div id="disease_graph"></div>
                    {{ disease_graph|safe }}
                </div>
                {% endif %}

                {% if wordcloud %}
                <div>
                    <div class="viz-title">Облако слов ключевых терминов</div>
                    <img src="data:image/png;base64,{{ wordcloud }}" alt="Word Cloud">
                </div>
                {% endif %}
            </div>
        </div>

        <h2>Список часто встречающихся тематических кластеров</h2>
        <ul>
            {% for topic in topics %}
                <li>{{ topic }}</li>
            {% endfor %}
        </ul>
    {% endif %}
    <a href="/" class="back-button">← Вернуться к поисковой строке</a>
</div>
</body></html>
"""

ABOUT_HTML = """
<!DOCTYPE html><html><head><title>О проекте</title>
<style>
    body { font-family: 'Segoe UI', Arial; margin: 20px; background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
    h1 { color: #333; text-align: center; margin-bottom: 30px; }
    p { font-size: 16px; line-height: 1.6; color: #333; }
    a { display: inline-block; margin-top: 20px; padding: 10px 20px; background: #007bff; color: white; border-radius: 5px; text-decoration: none; }
</style></head>
<body>
<div class="container">
    <h1>Описание проекта</h1>
    <p>Сайт призван помогать осуществлять семантический поиск по международным базам знаний и тезаурусам сотрудникам, преподавателям и студентам университета в рамках образовательного проекта на базе университета АНО РосНОУ.</p>
    <p>Цель образовательного проекта: <strong>использовать синергию OCR – технологий (например, Google Cloud Vision OCR, ABBYY Cloud OCR SDK) с большими языковыми моделями (LLM) и модифицированным алгоритмом LDA для обработки мультимодальных административных документов, содержащих текст, изображения и таблицы.</strong></p>
    <a href="/">← Назад на главную</a>
</div>
</body></html>
"""

# Text processing functions
lemmatizer = WordNetLemmatizer()
stop_words_base = set(stopwords.words('english')) | EXCLUDE_TERMS  # Base stop words


def detect_language(text):
    """Detect the language of the text with error handling."""
    if not text or not isinstance(text, str) or not text.strip():
        return 'en'
    try:
        return detect(text)
    except:
        return 'en'


def preprocess_text(text, lang='en', additional_stop_words=None):
    stop_words = stop_words_base.copy()
    if additional_stop_words:
        stop_words.update(additional_stop_words)
    if not text or not isinstance(text, str) or not text.strip():
        return []
    try:
        nlp = nlp_models.get(lang, nlp_models.get('en'))
        if not nlp:
            return []
        doc = nlp(text.lower())
        tokens = [lemmatizer.lemmatize(token.text) for token in doc
                  if token.is_alpha and token.text not in stop_words and len(token.text) > 2]
        return list(set(tokens))
    except Exception as e:
        logger.error(f"Error in preprocess_text: {e}")
        return []


def extract_potential_medical_terms(text, lang='en', additional_stop_words=None):
    stop_words = stop_words_base.copy()
    if additional_stop_words:
        stop_words.update(additional_stop_words)
    if not text or not isinstance(text, str) or not text.strip():
        return []
    try:
        nlp = nlp_models.get(lang, nlp_models.get('en'))
        if not nlp:
            return []
        doc = nlp(text.lower())
        terms = set()

        # Extract noun phrases and entities
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if len(chunk_text) > 2 and not any(w in stop_words for w in chunk_text.split()):
                terms.add(chunk_text)

        # Add single nouns
        for token in doc:
            if token.pos_ == 'NOUN' and len(token.text) > 2 and token.text not in stop_words:
                terms.add(token.lemma_)

        return list(terms)[:15]
    except Exception as e:
        logger.error(f"Error extracting terms: {e}")
        return []


def markup_text(text, lang='en'):
    if not text or not isinstance(text, str) or not text.strip():
        return text, []
    try:
        nlp = nlp_models.get(lang, nlp_models.get('en'))
        if not nlp:
            return text, []
        doc = nlp(text)
        marked_text = text
        markup_data = []

        for ent in doc.ents[:10]:
            if ent.label_ not in {'GPE', 'LOC', 'DATE', 'QUANTITY', 'MONEY'}:
                marked_text = marked_text.replace(ent.text, f'<span class="entity">{ent.text}</span>')
                markup_data.append({'type': ent.label_, 'text': ent.text})

        return marked_text, markup_data
    except Exception as e:
        logger.error(f"Error in markup_text: {e}")
        return text, []


def extract_key_entities(text, lang='en'):
    entities = {'persons': set(), 'geo_locations': set(), 'phone_numbers': set(), 'conditions': set(), 'rules': set(),
                'new_terms': set(), 'references': set(), 'article_goals': set(), 'article_methods': set(),
                'article_tools': set()}

    try:
        # Use SpaCy for general NER
        nlp = nlp_models.get(lang, nlp_models.get('en'))
        if nlp:
            doc = nlp(text)

            # Persons, Geo from SpaCy NER
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    entities['persons'].add(ent.text)
                elif ent.label_ in {'GPE', 'LOC'}:
                    entities['geo_locations'].add(ent.text)

            # Custom matcher for conditions, rules, references, goals, methods, tools
            matcher = Matcher(nlp.vocab)

            condition_patterns = [[{"LOWER": {"IN": ["must", "shall", "requires", "condition"]}}, {"POS": "VERB"}]]
            matcher.add("CONDITION", condition_patterns)

            rule_patterns = [[{"LOWER": {"IN": ["rule", "policy", "regulation"]}}]]
            matcher.add("RULE", rule_patterns)

            reference_patterns = [
                [{"LIKE_URL": True}],
                [{"TEXT": {"REGEX": r"\[\d+\]"}}],
                [{"TEXT": {"REGEX": r"\(\w+,\s*\d{4}\)"}}]
            ]
            matcher.add("REFERENCE", reference_patterns)

            goal_patterns = [[{"LOWER": {"IN": ["objective", "purpose", "aim", "goal"]}}]]
            matcher.add("GOAL", goal_patterns)

            method_patterns = [[{"LOWER": {"IN": ["method", "approach", "procedure", "technique"]}}]]
            matcher.add("METHOD", method_patterns)

            tool_patterns = [[{"LOWER": {"IN": ["tool", "instrument", "software", "device"]}}]]
            matcher.add("TOOL", tool_patterns)

            matches = matcher(doc)
            for match_id, start, end in matches:
                string_id = nlp.vocab.strings[match_id]
                span = doc[start:end]
                if string_id == "CONDITION":
                    entities['conditions'].add(span.text)
                elif string_id == "RULE":
                    entities['rules'].add(span.text)
                elif string_id == "REFERENCE":
                    entities['references'].add(span.text)
                elif string_id == "GOAL":
                    entities['article_goals'].add(span.text)
                elif string_id == "METHOD":
                    entities['article_methods'].add(span.text)
                elif string_id == "TOOL":
                    entities['article_tools'].add(span.text)

            # New terms: terms without WordNet synsets (previously unseen/rare)
            new_terms = []
            for token in doc:
                if token.is_alpha and len(token.text) > 3 and not wordnet.synsets(
                        token.text) and token.text not in stop_words_base:
                    entities['new_terms'].add(token.text)
                    new_terms.append(token.text)

            # Log rare terms for Nanonets annotation/training
            if new_terms:
                with open(RARE_TERMS_FILE, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(new_terms) + '\n')
                logger.info(f"Logged {len(new_terms)} rare terms for Nanonets training.")

        # Use Pullenti for additional NER
        analysis = SourceOfAnalysis(text)
        ar = pullenti_processor.process(analysis)
        if ar and ar.referents:
            for referent in ar.referents:
                if referent.type_name == "PERSON":
                    entities['persons'].add(str(referent))
                elif referent.type_name == "GEO":
                    entities['geo_locations'].add(str(referent))
                elif referent.type_name == "PHONE":
                    entities['phone_numbers'].add(str(referent))

        # Phone numbers fallback with regex
        phone_regex = re.findall(r'[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}', text)
        entities['phone_numbers'].update(phone_regex)

    except Exception as e:
        logger.error(f"Error in extract_key_entities: {e}")

    filtered_entities = {k: list(v) for k, v in entities.items() if v}

    # Check if extraction is insufficient (e.g., fewer than 3 total entities)
    total_entities = sum(len(v) for v in filtered_entities.values())
    if total_entities < 3:
        nanonets_results = nanonets_extract_entities(text)
        if nanonets_results:
            for k, v in nanonets_results.items():
                if k in filtered_entities:
                    filtered_entities[k].extend(v)
                else:
                    filtered_entities[k] = v
            logger.info("Merged Nanonets entities for better coverage.")
        else:
            abbyy_results = abbyy_extract_entities(text)
            if abbyy_results:
                for k, v in abbyy_results.items():
                    if k in filtered_entities:
                        filtered_entities[k].extend(v)
                    else:
                        filtered_entities[k] = v
                logger.info("Merged ABBYY entities as secondary fallback.")

    return filtered_entities


def nanonets_extract_entities(text):
    if not NANONETS_API_KEY or not NANONETS_MODEL_ID:
        logger.warning("Nanonets API key or model ID not set.")
        return {}
    try:
        response = requests.post(
            f"https://app.nanonets.com/api/v2/OCR/Model/{NANONETS_MODEL_ID}/LabelText/",
            headers={"Authorization": f"Basic {base64.b64encode((NANONETS_API_KEY + ':').encode()).decode()}"},
            data={"text": text}
        )
        if response.status_code == 200:
            data = response.json()
            entities = {}  # Parse based on Nanonets response for entities, keywords, topic
            # Assume custom model returns {'entities': {...}, 'keywords': [...], 'topic': 'string'}
            if 'result' in data:
                # Example parsing; adjust based on your model
                pred = data['result'][0].get('prediction', [])
                for p in pred:
                    if p['label'] in entities:
                        entities[p['label']].append(p['ocr_text'])
                    else:
                        entities[p['label']] = [p['ocr_text']]
            return entities
        else:
            logger.error(f"Nanonets error: {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Nanonets extraction error: {e}")
        return {}


def abbyy_extract_entities(text):
    if not ABBYY_APP_ID or not ABBYY_PASSWORD:
        logger.warning("ABBYY credentials not set.")
        return {}
    try:
        auth = base64.b64encode(f"{ABBYY_APP_ID}:{ABBYY_PASSWORD}".encode()).decode()
        response = requests.post(
            "https://cloud.ocrsdk.com/processTextField",
            headers={"Authorization": f"Basic {auth}"},
            data={"text": text}
        )
        if response.status_code == 200:
            data = response.json()
            entities = {}  # Parse ABBYY response for entities
            # Adjust parsing as per ABBYY response structure
            return entities
        else:
            logger.error(f"ABBYY error: {response.text}")
            return {}
    except Exception as e:
        logger.error(f"ABBYY extraction error: {e}")
        return {}


def extract_diseases(text, lang='en'):
    terms = extract_potential_medical_terms(text, lang)
    diseases = []
    for term in terms:
        if term in term_cache and term_cache[term] == 'disease':
            diseases.append(term)
            continue
        try:
            response = requests.get(
                "https://uts-ws.nlm.nih.gov/rest/search/current",
                params={"string": term, "apiKey": UMLS_API_KEY, "sabs": ",".join(prio_sabs)}
            )
            data = response.json()
            if 'result' in data and data['result']['results']:
                cui = data['result']['results'][0]['ui']
                if cui != 'NONE':
                    resp = requests.get(
                        f"https://uts-ws.nlm.nih.gov/rest/content/current/CUI/{cui}",
                        params={"apiKey": UMLS_API_KEY}
                    )
                    d = resp.json()
                    sem_types = [st['name'] for st in d['result']['semanticTypes']]
                    if 'Disease or Syndrome' in sem_types or 'Neoplastic Process' in sem_types:
                        diseases.append(term)
                        term_cache[term] = 'disease'
                    else:
                        term_cache[term] = 'not_disease'
                else:
                    term_cache[term] = 'not_disease'
            else:
                term_cache[term] = 'not_disease'
        except Exception as e:
            logger.error(f"UMLS query error for {term}: {e}")
    with open(TERM_CACHE_FILE, 'wb') as f:
        pickle.dump(term_cache, f)
    return diseases


def perform_enhanced_topic_modeling(documents):
    if not documents or not any(d and d.strip() for d in documents if isinstance(d, str)):
        return ["General Medical"] * len(documents) if documents else []

    topics = []
    valid_docs = [d for d in documents if isinstance(d, str) and d.strip()]

    if not valid_docs:
        return ["General Medical"] * len(documents)

    try:
        if len(valid_docs) >= 2:
            model = BERTopic(nr_topics=min(5, len(valid_docs)),
                             language="multilingual",
                             min_topic_size=1)
            topic_ids, _ = model.fit_transform(valid_docs)

            for doc in documents:
                if isinstance(doc, str) and doc.strip():
                    idx = valid_docs.index(doc)
                    if idx < len(topic_ids) and topic_ids[idx] != -1:
                        try:
                            topic_words = model.get_topic(topic_ids[idx])
                            if topic_words:
                                words = [word for word, _ in topic_words[:3]]
                                topics.append(f"{', '.join(words)}")
                            else:
                                topics.append("Medical Research")
                        except:
                            topics.append("Medical Research")
                    else:
                        topics.append("Medical Research")
                else:
                    topics.append("General Medical")

        else:
            for doc in documents:
                if isinstance(doc, str) and doc.strip():
                    terms = extract_potential_medical_terms(doc)
                    if terms:
                        topics.append(f"{', '.join(terms[:3])}")
                    else:
                        topics.append("Medical Research")
                else:
                    topics.append("General Medical")

    except Exception as e:
        logger.error(f"Error in topic modeling: {e}")
        for doc in documents:
            if isinstance(doc, str) and doc.strip():
                if "covid" in doc.lower() or "pandemic" in doc.lower():
                    topics.append("COVID-19 Research")
                elif "cancer" in doc.lower() or "tumor" in doc.lower():
                    topics.append("Oncology")
                elif "heart" in doc.lower() or "cardiac" in doc.lower():
                    topics.append("Cardiology")
                elif "brain" in doc.lower() or "neuro" in doc.lower():
                    topics.append("Neurology")
                else:
                    topics.append("General Medicine")
            else:
                topics.append("General Medical")

    return topics


def create_cluster_visualization(documents, topics):
    if not documents or not topics or sentence_model is None:
        return ""

    try:
        valid_docs = [d for d in documents if isinstance(d, str) and d.strip()]
        if len(valid_docs) < 2:
            return ""

        embeddings = sentence_model.encode(valid_docs)

        if len(valid_docs) >= 3:
            pca = PCA(n_components=min(3, len(valid_docs)))
            coords = pca.fit_transform(embeddings)
            if coords.shape[1] < 3:
                coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])), 'constant')
        else:
            coords = embeddings[:, :3] if embeddings.shape[1] >= 3 else np.pad(embeddings, ((0, 0), (0, 3)), 'constant')

        fig = go.Figure(data=[go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(coords)),
            mode='markers+text',
            marker=dict(
                size=10,
                color=list(range(len(topics))),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Topics")
            ),
            text=[t[:20] + "..." if len(t) > 20 else t for t in topics],
            textposition="top center",
            hovertemplate='<b>Topic:</b> %{text}<br><b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Z:</b> %{z}'
        )])

        fig.update_layout(
            title="Document Clusters in 3D Space",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
            height=500
        )

        return fig.to_html(div_id="cluster_plot")

    except Exception as e:
        logger.error(f"Error creating cluster visualization: {e}")
        return ""


def create_terms_timeline(results):
    if not results:
        return ""

    try:
        year_terms = defaultdict(set)

        for r in results:
            year_str = r.get('DP', '')
            if year_str and year_str != 'N/A':
                try:
                    year = int(year_str.split()[0]) if ' ' in year_str else int(year_str)
                    abstract = r.get('AB', '')
                    if isinstance(abstract, str) and abstract:
                        terms = extract_potential_medical_terms(abstract)
                        year_terms[year].update(terms[:5])
                except:
                    pass

        if not year_terms:
            return ""

        sorted_years = sorted(year_terms.keys())

        seen_terms = set()
        new_terms_data = []
        cumulative_terms = []

        for year in sorted_years:
            new_in_year = year_terms[year] - seen_terms
            new_terms_data.append(len(new_in_year))
            seen_terms.update(year_terms[year])
            cumulative_terms.append(len(seen_terms))

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("New Unique Terms per Year", "Cumulative Unique Terms"),
            vertical_spacing=0.15
        )

        fig.add_trace(
            go.Bar(x=sorted_years, y=new_terms_data, name="New Terms",
                   marker_color='indianred'),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=sorted_years, y=cumulative_terms, name="Cumulative",
                       mode='lines+markers', marker_color='lightseagreen'),
            row=2, col=1
        )

        fig.update_layout(height=500, showlegend=False,
                          title_text="Terms Evolution Over Time")
        fig.update_xaxes(title_text="Year", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)

        return fig.to_html(div_id="terms_timeline")

    except Exception as e:
        logger.error(f"Error creating terms timeline: {e}")
        return ""


def create_venn_diagram(topics):
    if not topics or len(topics) < 2:
        return ""

    try:
        topic_groups = defaultdict(set)

        for i, topic in enumerate(topics):
            if topic and topic != "General Medical":
                terms = topic.lower().split(',')[:3]
                for term in terms:
                    topic_groups[term.strip()].add(i)

        if len(topic_groups) < 2:
            return ""

        top_topics = sorted(topic_groups.items(), key=lambda x: len(x[1]), reverse=True)[:3]

        plt.figure(figsize=(10, 8))
        plt.clf()

        if len(top_topics) == 2:
            sets = [top_topics[0][1], top_topics[1][1]]
            venn = venn2(sets, set_labels=(top_topics[0][0][:20], top_topics[1][0][:20]))
        else:
            sets = [top_topics[0][1], top_topics[1][1], top_topics[2][1]]
            venn = venn3(sets, set_labels=(top_topics[0][0][:15],
                                           top_topics[1][0][:15],
                                           top_topics[2][0][:15]))

        plt.title("Topic Overlap Analysis")

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)  # Higher DPI for clarity
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Error creating Venn diagram: {e}")
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
            if t and t != "General Medical":
                topic_simple = t.split(',')[0] if ',' in t else t[:30]
                year_topic[y][topic_simple] += 1

        if not year_topic:
            return ""

        sorted_years = sorted(year_topic.keys())
        all_topics = list(set(t for c in year_topic.values() for t in c))[:10]

        plt.figure(figsize=(14, 8))
        plt.clf()

        matrix = np.zeros((len(all_topics), len(sorted_years)))
        for i, y in enumerate(sorted_years):
            for j, t in enumerate(all_topics):
                matrix[j, i] = year_topic[y].get(t, 0)

        colors = plt.cm.tab20(np.linspace(0, 1, len(all_topics)))  # Improved color palette

        plt.stackplot(sorted_years, matrix, labels=all_topics, colors=colors, alpha=0.8)
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=9)
        plt.title("Topic Distribution Over Time", fontsize=14, fontweight='bold')
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Number of Articles", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()

        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Error in plot_topic_frequency: {e}")
        return ""


def create_disease_article_graph(results, year_start, year_end):
    if not results or len(results) < 2 or sentence_model is None:
        return ""

    try:
        filtered_results = []
        for r in results:
            try:
                year = int(r.get('DP', '0').split()[0])
                if year_start <= year <= year_end:
                    filtered_results.append(r)
            except:
                pass

        if not filtered_results:
            return ""

        documents = [re.sub('<[^<]+?>', '', r.get('AB', '')) or r.get('TI', '') for r in filtered_results]
        embeddings = sentence_model.encode(documents)
        sim = cosine_similarity(embeddings)

        G = nx.Graph()
        for i, r in enumerate(filtered_results):
            label = f"{r.get('TI', 'No title')[:20]}...\n{r.get('AU', ['N/A'])[0]}\n{r.get('DP', 'N/A')}"
            G.add_node(i, label=label, diseases=r.get('diseases', []))

        for i in range(len(filtered_results)):
            for j in range(i + 1, len(filtered_results)):
                if sim[i][j] > 0.6:  # Lowered threshold for more connections
                    G.add_edge(i, j)

        pos = nx.spring_layout(G)

        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        node_x, node_y, node_text, node_color = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['label'])
            node_color.append(len(G.nodes[node]['diseases']))

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', hoverinfo='text', text=node_text,
                                marker=dict(showscale=True, colorscale='YlGnBu', size=10, color=node_color,
                                            colorbar=dict(thickness=15, title='Diseases Count')))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(showlegend=False, hovermode='closest',
                                         margin=dict(b=20, l=5, r=5, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         height=500))

        return fig.to_html(div_id="disease_graph")

    except Exception as e:
        logger.error(f"Error creating disease graph: {e}")
        return ""


def create_wordcloud(results):
    if not results:
        return ""
    try:
        text = ' '.join(' '.join(r.get('KW', [])) for r in results)
        if not text:
            return ""
        wc = WordCloud(background_color="white", max_words=100, colormap='plasma')  # Improved with colormap
        wc.generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error creating wordcloud: {e}")
        return ""


def search_core(query):
    try:
        headers = {"Authorization": f"Bearer {CORE_API_KEY}"}
        params = {"q": query, "limit": 10}
        response = requests.get("https://api.core.ac.uk/v3/search/works",
                                headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = []

        for item in data.get('results', []):
            abstract = item.get('abstract', '')
            keywords = item.get('keywords', [])

            if not keywords and abstract:
                keywords = extract_potential_medical_terms(abstract)[:5]

            result = {
                'TI': item.get('title', 'No title'),
                'DP': str(item.get('yearPublished', 'N/A')),
                'AB': abstract if abstract else 'No abstract available',
                'AU': item.get('authors', []),
                'KW': keywords if keywords else [],
                'LA': detect_language(abstract if abstract else item.get('title', '')),
                'Source': 'CORE',
                'ID': item.get('id')  # For recommend
            }
            results.append(result)

        return results
    except Exception as e:
        logger.error(f"CORE API error: {e}")
        return []


def get_similar_core(article_id):
    try:
        headers = {"Authorization": f"Bearer {CORE_API_KEY}", "Content-Type": "application/json"}
        body = {"articles": [{"id": article_id}], "limit": 3}
        response = requests.post("https://api.core.ac.uk/v3/recommend", headers=headers, json=body, timeout=10)
        response.raise_for_status()
        data = response.json()
        recommended = data.get('recommended', [])
        return [{'title': item.get('title', 'No title'),
                 'link': item.get('downloadUrl', '') or f"https://core.ac.uk/works/{item.get('id')}"}
                for item in recommended]
    except Exception as e:
        logger.error(f"CORE recommend error: {e}")
        return []


def get_similar_pubmed(pmid):
    try:
        handle = Entrez.elink(db="pubmed", id=pmid, cmd="neighbor", retmax=3)
        record = Entrez.read(handle)
        handle.close()
        if record and record[0].get('LinkSetDb'):
            similar_pmids = [link['Id'] for link in record[0]['LinkSetDb'][0]['Link']]
            if similar_pmids:
                handle = Entrez.efetch(db="pubmed", id=",".join(similar_pmids), rettype="medline", retmode="text")
                records = list(Medline.parse(StringIO(handle.read())))
                handle.close()
                return [{'title': r.get('TI', 'No title'),
                         'link': f"https://pubmed.ncbi.nlm.nih.gov/{similar_pmids[i]}"}
                        for i, r in enumerate(records)]
        return []
    except Exception as e:
        logger.error(f"PubMed similar error: {e}")
        return []


def search_similar_articles(query):
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=3)
        record = Entrez.read(handle)
        handle.close()
        pmids = record["IdList"]

        if pmids:
            handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
            records = list(Medline.parse(StringIO(handle.read())))
            handle.close()

            return [{'title': r.get('TI', 'No title'),
                     'link': f"https://pubmed.ncbi.nlm.nih.gov/{pmids[i]}"}
                    for i, r in enumerate(records) if r.get('TI')]
        return []
    except Exception as e:
        logger.error(f"Similar articles fallback error: {e}")
        return []


@app.route('/')
def home():
    return render_template_string(INDEX_HTML)


@app.route('/about')
def about():
    return render_template_string(ABOUT_HTML)


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '')
    db = request.form.get('db', 'pubmed')
    year_start = request.form.get('year_start', '2015')
    year_end = request.form.get('year_end', '2025')
    exclude_terms_str = request.form.get('exclude_terms', '')
    additional_exclude = [e.strip() for e in exclude_terms_str.split(',') if e.strip()]

    # Update stop words for this request
    stop_words = stop_words_base | set(additional_exclude)

    # Multilingual support: Translate query if not English for PubMed
    lang = detect_language(query)
    translated_query = query
    if t5_model and lang != 'en':
        input_text = f"translate to English: {query}"
        inputs = t5_tokenizer(input_text, return_tensors="pt")
        outputs = t5_model.generate(inputs, max_length=100)
        translated_query = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    if db == 'pubmed':
        search_query = translated_query
    else:
        search_query = query  # CORE supports multilingual

    results = []
    topics = []

    try:
        year_start = int(year_start)
        year_end = int(year_end)
        if year_start > year_end or year_start < 1900 or year_end > 2025:
            raise ValueError("Invalid year range")
    except:
        year_start, year_end = 2015, 2025

    if db == 'pubmed':
        try:
            handle = Entrez.esearch(db="pubmed", term=search_query, retmax=20)
            record = Entrez.read(handle)
            handle.close()
            pmids = record["IdList"]

            if pmids:
                handle = Entrez.efetch(db="pubmed", id=",".join(pmids),
                                       rettype="medline", retmode="text")
                records = list(Medline.parse(StringIO(handle.read())))
                handle.close()

                for i, r in enumerate(records):
                    year_str = r.get('DP', '')
                    if year_str:
                        try:
                            year = int(year_str.split()[0])
                            if year_start <= year <= year_end:
                                keywords = r.get('MH', [])
                                if not keywords:
                                    abstract = r.get('AB', '')
                                    if abstract:
                                        keywords = extract_potential_medical_terms(abstract,
                                                                                   additional_stop_words=additional_exclude)[
                                                   :8]

                                r['KW'] = keywords
                                r['Source'] = 'PubMed'
                                r['LA'] = detect_language(r.get('AB', ''))
                                r['PMID'] = pmids[i]

                                abstract = r.get('AB', '')
                                if abstract:
                                    lang = r['LA']
                                    marked_abstract, _ = markup_text(abstract, lang)
                                    r['AB'] = marked_abstract
                                    r['entities'] = extract_key_entities(abstract, lang)
                                    r['diseases'] = extract_diseases(abstract, lang)

                                    r['similar_articles'] = get_similar_pubmed(r['PMID'])[:2]
                                else:
                                    r['AB'] = 'No abstract available'
                                    r['entities'] = {}
                                    r['diseases'] = []
                                    r['similar_articles'] = []

                                results.append(r)
                        except:
                            pass
        except Exception as e:
            logger.error(f"PubMed error: {e}")

    else:  # CORE
        core_results = search_core(search_query)
        for r in core_results:
            try:
                year_str = r.get('DP', '0')
                if year_str != 'N/A':
                    year = int(year_str)
                    if year_start <= year <= year_end:
                        abstract = r.get('AB', '')
                        if abstract and abstract != 'No abstract available':
                            lang = r['LA']
                            marked_abstract, _ = markup_text(abstract, lang)
                            r['AB'] = marked_abstract
                            r['entities'] = extract_key_entities(abstract, lang)
                            r['diseases'] = extract_diseases(abstract, lang)
                            r['similar_articles'] = get_similar_core(r['ID'])[:2]
                        else:
                            r['entities'] = {}
                            r['diseases'] = []
                            r['similar_articles'] = []
                        results.append(r)
                else:
                    results.append(r)
            except:
                results.append(r)

    visualizations = {}

    if results:
        documents = []
        for r in results:
            abstract = r.get('AB', '')
            if abstract:
                clean_abstract = re.sub('<[^<]+?>', '', abstract)
                documents.append(clean_abstract)
            else:
                documents.append(r.get('TI', ''))

        topics = perform_enhanced_topic_modeling(documents)

        for i, result in enumerate(results):
            if i < len(topics):
                result['Topic'] = topics[i]
            else:
                result['Topic'] = "Medical Research"

        visualizations['year_plot'] = plot_topic_frequency(results, topics)
        visualizations['cluster_plot'] = create_cluster_visualization(documents, topics)
        visualizations['terms_timeline'] = create_terms_timeline(results)
        visualizations['venn_plot'] = create_venn_diagram(topics)
        visualizations['disease_graph'] = create_disease_article_graph(results, year_start, year_end)
        visualizations['wordcloud'] = create_wordcloud(results)

    if not results:
        results = [{'TI': 'No results found', 'DP': 'N/A', 'AB': '',
                    'AU': [], 'KW': [], 'LA': 'en', 'Source': db,
                    'Topic': 'N/A', 'entities': {}, 'diseases': [], 'similar_articles': []}]
        topics = []

    return render_template_string(RESULTS_HTML,
                                  results=results,
                                  db=db,
                                  topics=list(set(topics)) if topics else [],
                                  year_start=year_start,
                                  year_end=year_end,
                                  **visualizations)


@app.route('/upload', methods=['POST'])
def upload_document():
    if 'file' not in request.files:
        return render_template_string(RESULTS_HTML,
                                      results=[{'TI': 'No file uploaded', 'DP': 'N/A',
                                                'AB': 'Please select a file', 'AU': [],
                                                'KW': [], 'LA': 'en', 'Source': 'Upload',
                                                'Topic': 'N/A', 'entities': {}, 'diseases': []}],
                                      db='upload', topics=[],
                                      year_start=2015, year_end=2025)

    file = request.files['file']
    text = process_document(file)

    if text:
        lang = detect_language(text)
        marked_text, _ = markup_text(text[:1000], lang)
        keywords = extract_potential_medical_terms(text, lang)[:10]
        entities = extract_key_entities(text, lang)
        diseases = extract_diseases(text, lang)

        topics = perform_enhanced_topic_modeling([text])

        search_terms = ' '.join(keywords[:3]) if keywords else text[:100]
        similar_articles = search_similar_articles(search_terms)[:2]

        results = [{
            'TI': f'Uploaded: {file.filename}',
            'DP': 'N/A',
            'AB': marked_text + ("..." if len(text) > 1000 else ""),
            'AU': [],
            'KW': keywords,
            'LA': lang,
            'Topic': topics[0] if topics else 'Document Analysis',
            'entities': entities,
            'diseases': diseases,
            'Source': 'Upload',
            'similar_articles': similar_articles
        }]
    else:
        results = [{'TI': 'Error processing file', 'DP': 'N/A',
                    'AB': 'Could not extract text from file', 'AU': [],
                    'KW': [], 'LA': 'en', 'Source': 'Upload',
                    'Topic': 'N/A', 'entities': {}, 'diseases': [], 'similar_articles': []}]
        topics = []

    return render_template_string(RESULTS_HTML,
                                  results=results,
                                  db='upload',
                                  topics=topics if text else [],
                                  year_start=2015,
                                  year_end=2025)


def preprocess_image_for_ocr(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            angle = rect[2]

            if angle < -45:
                angle = -(90 + angle)
            if abs(angle) > 5:
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None


def process_document(file):
    if not file:
        return ""
    text = ""
    filename = file.filename
    temp_path = f'temp_{filename}'
    file.save(temp_path)
    try:
        if filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(temp_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif filename.endswith('.docx'):
            text = docx2txt.process(temp_path)
        elif filename.endswith('.xlsx'):
            workbook = openpyxl.load_workbook(temp_path)
            sheet = workbook.active
            text = "\n".join(" ".join(str(cell.value) for cell in row if cell.value) for row in sheet.rows)
        elif filename.endswith(('.png', '.jpg', '.tiff')):
            preprocessed_img = preprocess_image_for_ocr(temp_path)
            if preprocessed_img is not None:
                reader = easyocr.Reader(['en', 'ru'], gpu=False)
                ocr_results = reader.readtext(preprocessed_img)
                text = ' '.join([res[1] for res in ocr_results])

        if text:
            lang = detect_language(text)
            entities = extract_key_entities(text, lang)
            total_entities = sum(len(v) for v in entities.values())
            if total_entities < 3:
                nanonets_text = nanonets_ocr(temp_path)
                if nanonets_text:
                    text = nanonets_text
                    logger.info("Fallback to Nanonets OCR successful.")
                else:
                    abbyy_text = abbyy_ocr(temp_path)
                    if abbyy_text:
                        text = abbyy_text
                        logger.info("Fallback to ABBYY OCR successful.")
    except Exception as e:
        logger.error(f"Error processing {filename}: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return text


def nanonets_ocr(file_path):
    if not NANONETS_API_KEY or not NANONETS_MODEL_ID:
        logger.warning("Nanonets credentials not set.")
        return ""
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"https://app.nanonets.com/api/v2/OCR/Model/{NANONETS_MODEL_ID}/LabelFile/",
                headers={"Authorization": f"Basic {base64.b64encode((NANONETS_API_KEY + ':').encode()).decode()}"},
                files={"file": f}
            )
        if response.status_code == 200:
            data = response.json()
            return data.get('result', [{}])[0].get('prediction', [{}])[0].get('ocr_text', '')
        else:
            logger.error(f"Nanonets OCR error: {response.text}")
            return ""
    except Exception as e:
        logger.error(f"Nanonets OCR error: {e}")
        return ""


def abbyy_ocr(file_path):
    if not ABBYY_APP_ID or not ABBYY_PASSWORD:
        logger.warning("ABBYY credentials not set.")
        return ""
    try:
        auth = base64.b64encode(f"{ABBYY_APP_ID}:{ABBYY_PASSWORD}".encode()).decode()
        with open(file_path, 'rb') as f:
            response = requests.post(
                "https://cloud.ocrsdk.com/processImage?exportFormat=txt",
                headers={"Authorization": f"Basic {auth}"},
                files={"file": f}
            )
        if response.status_code == 200:
            task_id = response.text
            for _ in range(10):
                status_resp = requests.get(f"https://cloud.ocrsdk.com/getTaskStatus?taskId={task_id}",
                                           headers={"Authorization": f"Basic {auth}"})
                if status_resp.status_code == 200 and 'Completed' in status_resp.text:
                    result_resp = requests.get(f"https://cloud.ocrsdk.com/getResult?taskId={task_id}",
                                               headers={"Authorization": f"Basic {auth}"})
                    return result_resp.text
                time.sleep(2)
            logger.error("ABBYY task timed out.")
            return ""
        else:
            logger.error(f"ABBYY OCR error: {response.text}")
            return ""
    except Exception as e:
        logger.error(f"ABBYY OCR error: {e}")
        return ""


if __name__ == '__main__':
    app.run(debug=True)