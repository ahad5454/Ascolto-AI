import os
import json
import streamlit as st
from datetime import datetime
from chatbot import CompanyChatbot
from dotenv import load_dotenv
from openai import OpenAI
from utils import chunk_text, get_openai_embedding
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0  # for consistent language detection results

st.set_page_config(
    page_title="Ascolto AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize persistent state
for key, default in {
    "messages": [],
    "saved_chats": [],
    "call_ready": False,
    "is_processing": False,
    "fully_cleared_on_start": False,
    "uploader_key": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

load_dotenv()

@st.cache_resource
def get_openai_client():
    return OpenAI()

def get_chatbot():
    return CompanyChatbot()

chatbot = get_chatbot()

def clear_single_call_artifacts(full_reset=True):
    """Clears embeddings, audio, transcripts, and resets states."""
    from database import EmbeddingDatabase
    try:
        EmbeddingDatabase("embeddings_db").clear_database()
    except Exception:
        pass

    for folder in ["audio", "transcripts", "calls"]:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                try:
                    os.rmdir(root)
                except Exception:
                    pass

    for k in ["processed_audio", "last_transcript_path", "last_audio_path", "last_call_id"]:
        st.session_state.pop(k, None)

    st.session_state.call_ready = False
    st.session_state.is_processing = False

    if full_reset:
        st.cache_resource.clear()
        st.session_state.pop("chatbot", None)

if not st.session_state.fully_cleared_on_start:
    clear_single_call_artifacts()
    st.session_state.fully_cleared_on_start = True

with st.sidebar:
    st.title("💬 Chat History / Cronologia Chat")

    if st.button("➕ New Chat / Nuova Chat"):
        if st.session_state.messages:
            st.session_state.saved_chats.append(list(st.session_state.messages))

        clear_single_call_artifacts(full_reset=True)
        st.session_state.messages = []
        st.session_state.chatbot = get_chatbot()
        st.session_state.uploader_key += 1

        st.success("All old data cleared. Please upload a new audio file. / Tutti i dati precedenti sono stati eliminati. Carica un nuovo file audio.")
        st.rerun()

    st.markdown("---")

    for i, chat in enumerate(st.session_state.saved_chats):
        if st.button(f"Chat {i+1}", key=f"chat_{i}") and not st.session_state.is_processing:
            st.session_state.messages = chat

    st.markdown("---")
    st.subheader("🎧 Upload Call Audio / Carica Audio della Chiamata")

    uploaded = st.file_uploader(
        "Upload an audio file / Carica un file audio",
        type=["mp3", "wav", "m4a", "mp4", "mpeg", "ogg", "webm", "aac", "flac"],
        accept_multiple_files=False,
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded is not None:
        if st.session_state.is_processing:
            st.warning("Processing in progress, please wait... / Elaborazione in corso, attendere prego...")
        elif st.session_state.call_ready:
            st.info("Start a new chat to upload another file. / Avvia una nuova chat per caricare un altro file.")
        else:
            st.session_state.is_processing = True
            st.session_state.call_ready = False

            with st.spinner("🔄 Clearing old data and processing new audio... / Pulizia dei vecchi dati e elaborazione del nuovo audio..."):
                try:
                    clear_single_call_artifacts(full_reset=False)
                    os.makedirs("audio", exist_ok=True)

                    audio_path = os.path.join("audio", uploaded.name)
                    with open(audio_path, "wb") as f:
                        f.write(uploaded.getbuffer())

                    client = get_openai_client()
                    with open(audio_path, "rb") as af:
                        result = client.audio.translations.create(
                            model="whisper-1",
                            file=af,
                            response_format="json",
                            temperature=0
                        )

                    transcript_text = getattr(result, "text", "").strip()
                    os.makedirs("transcripts", exist_ok=True)
                    base = os.path.splitext(uploaded.name)[0]
                    transcript_path = os.path.join("transcripts", f"{base}.txt")
                    with open(transcript_path, "w", encoding="utf-8") as tf:
                        tf.write(transcript_text)

                    from database import EmbeddingDatabase
                    db = EmbeddingDatabase("embeddings_db")
                    db.clear_database()
                    call_id = "current_call"

                    chunks = chunk_text(transcript_text)
                    embeddings_data = []
                    for idx, ch in enumerate(chunks):
                        emb = get_openai_embedding(ch)
                        embeddings_data.append({
                            "text": ch,
                            "embedding": emb,
                            "metadata": {
                                "pdf_name": call_id,
                                "page_num": idx + 1,
                                "call_id": call_id,
                                "chunk_index": idx
                            }
                        })
                    db.add_pdf_embeddings(call_id, embeddings_data)

                    st.session_state.update({
                        "processed_audio": uploaded.name,
                        "last_transcript_path": transcript_path,
                        "last_audio_path": audio_path,
                        "last_call_id": call_id,
                        "call_ready": True,
                        "is_processing": False
                    })

                    st.success(f"Audio processed successfully — {len(embeddings_data)} chunks indexed. / Audio elaborato con successo — {len(embeddings_data)} segmenti indicizzati.")
                    st.caption("📄 Transcript preview / Anteprima della trascrizione:")
                    st.code(transcript_text[:2000] + ("..." if len(transcript_text) > 2000 else ""))

                except Exception as e:
                    st.session_state.is_processing = False
                    st.error(f"Processing failed: {e} / Elaborazione fallita: {e}")

# ---------------- Main Chat Area ---------------- #
st.title("🤖 Ascolto AI")
st.caption("Ask questions about your uploaded call audio — Rispondi alle domande sulla tua chiamata.")

if st.session_state.is_processing:
    st.warning("⏳ Please wait — current audio is still processing. / Attendere prego — l'audio è ancora in elaborazione.")
    st.stop()

if not st.session_state.call_ready:
    st.warning("Please upload and process a new audio file before asking questions. / Carica ed elabora un nuovo file audio prima di fare domande.")
    st.stop()

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f"{msg['content']}\n\n*{msg.get('timestamp', '')}*")

# ---- Improved language detection ---- #
def detect_language(text: str) -> str:
    """Detect the language using langdetect with fallback to heuristic."""
    try:
        lang = detect(text)
        if lang.startswith("it"):
            return "it"
        elif lang.startswith("en"):
            return "en"
        else:
            return "mixed"
    except LangDetectException:
        lower = text.lower()
        if any(word in lower for word in ["ciao", "chiamata", "grazie", "questo", "signora", "signor"]):
            return "it"
        elif any(word in lower for word in ["call", "about", "who", "customer", "agent"]):
            return "en"
        else:
            return "mixed"

# ---- Handle new question ---- #
if prompt := st.chat_input("💭 Ask a question about the uploaded call... / Fai una domanda sulla chiamata caricata..."):
    if st.session_state.is_processing or not st.session_state.call_ready:
        st.warning("Please wait until audio processing finishes before asking questions. / Attendere che l'elaborazione audio termini prima di porre domande.")
        st.stop()

    user_lang = detect_language(prompt)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    with st.chat_message("user"):
        st.markdown(f"{prompt}\n\n*{timestamp}*")

    with st.spinner("💡 Thinking... / Sto pensando..."):
        try:
            response = chatbot.ask_question(prompt)
        except Exception as e:
            response = {"answer_en": f"Error: {e}", "answer_it": f"Errore: {e}", "sources": []}

    english_answer = response.get("answer_en", "").strip()
    italian_answer = response.get("answer_it", "").strip()
    sources = response.get("sources", [])

    # Output format logic
    if user_lang == "it":
        display_text = f"🇮🇹 **Italiano:** {italian_answer or english_answer}"
    elif user_lang == "en":
        display_text = f"**English:** {english_answer}\n\n🇮🇹 **Italiano:** {italian_answer}"
    else:  # mixed or uncertain
        display_text = f"**English:** {english_answer}\n\n🇮🇹 **Italiano:** {italian_answer}"

    if sources:
        display_text += "\n\n**Referenced Audio(s) / Audio di riferimento:** " + ", ".join(sources)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.messages.append({
        "role": "assistant",
        "content": display_text,
        "timestamp": timestamp
    })

    with st.chat_message("assistant"):
        st.markdown(f"{display_text}\n\n*{timestamp}*")
