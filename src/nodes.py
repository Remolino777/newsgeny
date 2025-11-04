from typing import Dict, Any, List
import requests
import os
import dotenv
import urllib
import json

from src.prompts import PROMPTS
from src.states import AgentInput, PrivateState, AgentOutput, OverallState, GradeQuestion

from langchain_core.documents import Document   #### Langchain.schema is deprecated for documents
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
NEWSDATA_KEY = os.getenv("NEWSDATA_KEY") or st.secrets["NEWSDATA_KEY"]
GNEWS_KEY = os.getenv("GNEWS_KEY") or st.secrets["GNEWS_KEY"]

import logging

logger = logging.getLogger(__name__)


def question_rewriter(state: AgentInput) -> PrivateState:
    rephrased_question = PROMPTS["rephrase"]
    print(f"Entering question_rewriter with following state: {state}")
    logger.info(f"Entering question_rewriter with following state: {state}")
    logger.info(f"Rephrased question: {rephrased_question}")

    prompt = PROMPTS["rephrase"].format(user_question=state['question'].content)
    # Reset state variables except for 'question' and 'messages'
    state["news"] = []
    state["reformulated_once"] = 0
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    if len(state["messages"]) > 1:
        conversation = state["messages"][:-1]
        current_question = state["question"].content
        messages = [
            SystemMessage(
    content=prompt
        )
        ]
        messages.extend(conversation)
        messages.append(HumanMessage(content=current_question))
        rephrase_prompt = ChatPromptTemplate.from_messages(messages)
        llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # o gemini-1.5-pro para más potencia
                google_api_key=GEMINI_API_KEY,
                    )
        prompt = rephrase_prompt.format()
        response = llm.invoke(prompt)
        better_question = response.content.strip()
        print(f"question_rewriter: Rephrased question: {better_question}")
        state["rephrased_question"] = better_question
    else:
        state["rephrased_question"] = state["question"].content
    return state

def question_classifier(state: PrivateState) -> PrivateState:
    print("Entering question_classifier")
    logger.info("Entering question_classifier")
    prompt_text = PROMPTS["classification"].format(
        user_question=state.get("rephrased_question", "")
    )
    system_message = SystemMessage(
    content=prompt_text
    )
    human_message = HumanMessage(
        content=f"User question: {state['rephrased_question']}"
    )
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # o gemini-1.5-pro para más potencia
                google_api_key=GEMINI_API_KEY,
                    )
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({"rephrased_question": state.get("rephrased_question", "")})
    if result is None:
        logger.warning("❌ question_classifier: result is None")
        state["on_topic"] = "No"
    else:
        state["on_topic"] = result.score.strip() if getattr(result, "score", None) else "No"

    logger.info(f"question_classifier: on_topic = {state['on_topic']}")
    return state

def safe_fetch_article(article_url: str, timeout: int = 10) -> str | None:
    """
    Intenta recuperar el HTML/texto del artículo con headers; devuelve None si falla o responde 403.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        resp = requests.get(article_url, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            return resp.text  # o procesa/extrae texto aquí
        else:
            print(f"❌ Fetch article {article_url} returned HTTP {resp.status_code}")
            return None
    except requests.RequestException as e:
        print(f"❌ Exception fetching article {article_url}: {e}")
        return None

def get_news_A_gnews(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Obtiene noticias desde GNews API usando el estado privado (PrivateState).
    Lee la query desde el estado y guarda los resultados en él.
    """
    API_KEY = GNEWS_KEY  # 🔑 tu clave GNews
    query = state.get("query", None)
    category = state.get("category", "general")
    max_results = state.get("max_results", 15)
    lang = state.get("lang", "en")
    country = state.get("country", "us")

    # 🧹 Limpieza de la query
    if query:
        query = query.strip().replace("?", "")
        query_encoded = urllib.parse.quote(query)
        url = f"https://gnews.io/api/v4/search?q={query_encoded}&lang={lang}&max={max_results}&apikey={API_KEY}"
    else:
        url = f"https://gnews.io/api/v4/top-headlines?category={category}&lang={lang}&country={country}&max={max_results}&apikey={API_KEY}"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))

            if "errors" in data:
                print("⚠️ Error desde GNews:", data["errors"])
                state["news"] = []
                return state

            articles = data.get("articles", [])
            if not articles:
                print("⚠️ No se encontraron artículos.")
                state["news"] = []
                return state

            news_docs = []
            for a in articles:
                content = a.get("description") or a.get("title", "")
                news_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "title": a.get("title"),
                            "url": a.get("url"),
                            "source": a.get("source", {}).get("name"),
                            "published_at": a.get("publishedAt"),
                            "image": a.get("image"),
                        },
                    )
                )

            # ✅ Guardamos los resultados dentro del estado
            state["news"] = news_docs

            print("\n🗞️ GNews results:")
            for i, doc in enumerate(news_docs, 1):
                print(f"{i}. {doc.metadata['title']} - {doc.metadata['url']}")

    except urllib.error.HTTPError as e:
        print(f"❌ Error HTTP {e.code}: {e.reason}")
        state["news"] = []
    except urllib.error.URLError as e:
        print(f"❌ Error de conexión: {e.reason}")
        state["news"] = []
    except json.JSONDecodeError:
        print("❌ Error al decodificar JSON.")
        state["news"] = []
    except Exception as e:
        print(f"⚠️ Error inesperado: {e}")
        state["news"] = []

    return state

def get_news_B_newsdata(state: Dict) -> Dict:
    """
    Node function for LangGraph.
    Fetches news from NewsData.io API using the query in the PrivateState
    and appends unique Document objects to `state["news"]`.
    """
    query = state.get("rephrased_question") or state.get("query", "")
    if not query:
        print("⚠️ No query provided.")
        return state

    api_key = NEWSDATA_KEY
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": api_key,
        "q": query,
        "language": "en",
        "country": "us",
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching news: {e}")
        return state

    articles = data.get("results", [])
    if not articles:
        print("⚠️ No relevant news found.")
        return state

    # 🔹 Evitar duplicados por título o contenido
    seen_titles = set()
    news_docs = []
    for a in articles:
        title = (a.get("title") or "").strip().lower()
        content = (a.get("description") or "").strip().lower()

        if not title:
            continue

        # Clave para detectar duplicados (misma noticia en distintos medios)
        key = (title, content[:100])  # primeros 100 chars del contenido
        if key in seen_titles:
            continue
        seen_titles.add(key)

        doc = Document(
            page_content=a.get("description") or a.get("title", ""),
            metadata={
                "title": a.get("title", "No title"),
                "url": a.get("link", ""),
                "source": a.get("source_id", ""),
                "published_at": a.get("pubDate", ""),
                "image": a.get("image_url", ""),
            },
        )
        news_docs.append(doc)

    # 🔹 Limitar a 5 noticias únicas por API
    news_docs = news_docs[:15]

    # Debug print
    print("\n🗞️ Unique NewsData.io results:")
    for i, doc in enumerate(news_docs, 1):
        print(f"{i}. {doc.metadata['title']} - {doc.metadata['source']}")

    # 🔹 Inicializar y agregar al estado
    if "news" not in state or not isinstance(state["news"], list):
        state["news"] = []
    state["news"].extend(news_docs)

    return state

def get_news(state: Dict) -> Dict:
    """
    Unified node function that orchestrates provider-specific news fetching.
    Each provider updates state["news"] directly.
    """
    query = state.get("rephrased_question") or state.get("query", "")
    if not query:
        print("⚠️ No query provided in state.")
        return state

    # ensure news exists
    if "news" not in state or not isinstance(state["news"], list):
        state["news"] = []

    # run provider nodes (each one appends to state["news"])    
    state = get_news_B_newsdata(state)
    state = get_news_A_gnews(state)

    print(f"✅ Total articles after fetch: {len(state['news'])}")
    return state

def combine_news(state: Dict) -> Dict:
    model =  ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",  # o gemini-1.5-pro para más potencia
                google_api_key=GEMINI_API_KEY,
                    )
    print("\n🧠 Entering combine_news_gemini node")
    prompt = PROMPTS['combine_news'].format(combine_news=state['news'])

    # Si ya reformuló una vez, marcamos como completado
    if state.get("reformulated_once", 0) == 1:
        print("🔁 Reformulation already done once. Formatting and ending news cycle.")
        all_news: List[Document] = state.get("news", [])
        print("\n🗞️ Latest News Found:")
        for doc in all_news:
            print(f"- {doc.metadata.get('title', 'No title')}: {doc.metadata.get('url', '')}")

        markdown_output = "### 🗞️ Latest News\n\n"
        for i, doc in enumerate(all_news, start=1):
            title = doc.metadata.get("title", "No title")
            url = doc.metadata.get("url", "")
            source = doc.metadata.get("source", "Unknown source")
            markdown_output += f"{i}. [{title}]({url}) — *{source}*\n"

        state["formatted_news_md"] = markdown_output.strip()
        state["news_cycle_done"] = True  # ✅ Aquí se detendrá el ciclo
        return state

    all_news: List[Document] = state.get("news", [])
    if not all_news:
        print("⚠️ No news found in state.")
        state["formatted_news_md"] = "No news available."
        state["reformulated_once"] = 0
        return state

    combined_text = "\n".join(
        [f"- {doc.metadata.get('title', 'No title')}: {doc.metadata.get('url', '')}" for doc in all_news]
    )
    response = model.invoke(prompt)
    result_text = response.content.strip()
    print("\n🤖 Gemini response:")
    print(result_text)

    if "NO DUPLICATES" in result_text.upper():
        markdown_output = "### 🗞️ Latest News\n\n"
        for i, doc in enumerate(all_news, start=1):
            title = doc.metadata.get("title", "No title")
            url = doc.metadata.get("url", "")
            source = doc.metadata.get("source", "Unknown source")
            markdown_output += f"{i}. [{title}]({url}) — *{source}*\n"
        state["formatted_news_md"] = markdown_output.strip()
        state["reformulated_once"] = 0
        state["news_cycle_done"] = True  # ✅ También termina el ciclo
        print("\n✅ No duplicates found. Markdown output ready.")
    else:
        state["reformulated_once"] = 1
        state["more_info_question"] = result_text
        state["news_cycle_done"] = False  # 👈 Aún no termina
        print("\n⚠️ Duplicates detected. Reformulating question...")

    return state

def off_topic_response(state: PrivateState) -> PrivateState:
    """
    Genera una respuesta cuando la pregunta no es de temas actuales.
    """
    logger.info("Entering off_topic_response")
    print("Entering off_topic_response")

    # Asegurar campo messages
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    # Crear mensaje tipo AIMessage
    ai_message = AIMessage(
        content="⚠️ Sorry, that question seems unrelated to news or real-time updates."
    )

    # Agregar al historial de mensajes
    state["messages"].append(ai_message)

    # Marcar banderas de control del flujo
    state["on_topic"] = "no"
    state["proceed_to_generate"] = False

    # Retornar el estado actualizado
    return state

def on_topic_response(state: PrivateState):
    print("Entering on_topic_response")

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    news_list = state.get("news", [])
    if not news_list:
        markdown_content = "⚠️ **There are no relevant news articles found.**"
    else:
        # 🔹 Eliminar duplicados por título o URL
        unique_news = []
        seen = set()
        for doc in news_list:
            title = doc.metadata.get("title", "").strip()
            url = doc.metadata.get("source", "").strip()  # 'source' o 'url' según tu schema
            key = (title.lower(), url.lower())
            if key not in seen:
                seen.add(key)
                unique_news.append(doc)

        # 🔹 Tomar solo las 10 primeras noticias
        top_news = unique_news[:10]

        # 🔹 Crear el contenido Markdown
        markdown_content = "## 📰 Top 10 News Articles\n\n"
        for i, doc in enumerate(top_news, start=1):
            title = doc.metadata.get("title", f"Article {i}")
            url = doc.metadata.get("source", "#")
            summary = doc.page_content.strip().split("\n")[0]
            markdown_content += f"**{i}. [{title}]({url})**\n\n> {summary}\n\n---\n"

        markdown_content += "\n✨ _These are the most relevant news articles at the moment._"

    # Añadir el mensaje formateado al estado
    state["messages"].append(AIMessage(content=markdown_content))
    return state