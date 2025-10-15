# file: company_chatbot_app.py
import logging
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import streamlit as st
from openai import OpenAI
from database import EmbeddingDatabase
from utils import get_openai_embedding
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="chatbot_log.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def extract_first_json(text: str) -> Optional[dict]:
    """
    Try to parse JSON from text. If direct json.loads fails,
    extract the first {...} block with balanced braces and parse it.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    break
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def translate_to_italian(english_text: str, model: str = "gpt-4-turbo") -> str:
    """
    Translate English text to Italian using the same chat completion interface.
    Returns Italian text or a fallback error message.
    """
    try:
        resp = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. Translate the user's text from English to Italian. "
                        "Keep tone natural and businesslike. Do not add commentary, metadata, or markup — only the translation."
                    )
                },
                {"role": "user", "content": english_text}
            ],
            temperature=0.0,
            max_tokens=800
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return "Errore durante la traduzione in italiano."


class CompanyChatbot:
    def __init__(self, db_path="embeddings_db", top_k=12):
        self.db = EmbeddingDatabase(db_path)
        self.model_extraction = "gpt-3.5-turbo"  # faster deterministic extraction
        self.model_final = "gpt-4-turbo"          # high-quality final output
        self.top_k = top_k
        self.log_dir = "chat_logs"
        os.makedirs(self.log_dir, exist_ok=True)

        # Extraction prompt (full schema)
        self.extraction_prompt = (
            "You are an expert Call Quality Analyst. Your task is to extract structured insights from a single call transcript.\n\n"
            "Focus on both content and performance aspects of the agent. Read the provided transcript carefully and OUTPUT STRICTLY VALID JSON ONLY using this schema exactly:\n\n"
            "{\n"
            '  "summary": "1-3 sentences summarizing the purpose and outcome of the call.",\n'
            '  "participants": [ { "name": "name or unknown", "role": "agent|customer|other|unknown" } ],\n'
            '  "call_roles": { "caller": "name or unknown", "callee": "name or unknown" },\n'
            '  "topics": ["main discussion topics"],\n'
            '  "qa_pairs": [ { "question": "...", "answer": "...", "asked_by": "agent|customer|unknown", "answered_by": "agent|customer|unknown", "evidence": ["short quote 1", "short quote 2"] } ],\n'
            '  "objections": ["list of customer concerns or rejections if any"],\n'
            '  "decisions": ["agreements, purchases, or commitments made during the call"],\n'
            '  "follow_ups": ["promises, next steps, or tasks agreed upon"],\n'
            '  "sentiment": "positive|neutral|negative|mixed",\n'
            '  "agent_performance": {\n'
            '      "clarity": "1-10 rating with a short reason",\n'
            '      "empathy": "1-10 rating with a short reason",\n'
            '      "professionalism": "1-10 rating with a short reason",\n'
            '      "listening_skills": "1-10 rating with a short reason",\n'
            '      "problem_resolution": "1-10 rating with a short reason",\n'
            '      "sales_effectiveness": "1-10 rating with a short reason (if sales context)"\n'
            '  },\n'
            '  "key_quotes": ["short impactful or emotional quotes from the call"],\n'
            '  "overall_call_quality": "1-10 overall performance rating with one-sentence justification"\n'
            "}\n\n"
            "Rules:\n"
            "- Use ONLY the given transcript context. Never fabricate information.\n"
            "- If data is unavailable, omit that field or use null.\n"
            "- Ratings should be numeric 1-10 and include a short reason.\n"
            "- Provide short verbatim quotes in key_quotes and any evidence fields.\n"
            "- Output MUST be valid JSON (no extra commentary)."
        )

        # Final user-facing response prompt
        self.final_prompt = (
            "You are a senior Call Quality Analyst. You will be given a structured JSON (from the extraction step) describing a single call.\n\n"
            "Use ONLY that JSON to answer the user's question. Do NOT invent facts or use outside knowledge.\n\n"
            "Capabilities:\n"
            "- Summarize the call (2-4 sentences).\n"
            "- Identify who spoke and roles.\n"
            "- Evaluate agent performance and explain using the agent_performance fields.\n"
            "- Provide short suggestions for improvement (bullet points).\n"
            "- Quote at most one short evidence snippet when helpful.\n\n"
            "Formatting rules:\n"
            "- Answer in natural language (no JSON) and be concise.\n"
            "- For evaluative questions, reference numeric scores and 1-line justification.\n"
            "- For factual questions, answer directly. If the JSON lacks the answer reply exactly: I couldn't find that information in the call.\n"
        )

    def ask_question(self, user_query: str) -> Dict:
        return self._ask(user_query)

    def _wrap_and_translate(self, english_text: str, sources: List[str]) -> Dict:
        italian = translate_to_italian(english_text, model=self.model_final)
        return {
            "answer_en": english_text,
            "answer_it": italian,
            "sources": sources
        }

    def _ask(self, user_query: str) -> Dict:
        # Step 0: Create embedding for query
        try:
            query_embedding = get_openai_embedding(user_query)
        except Exception as e:
            logging.error(f"Embedding error: {e}")
            return self._wrap_and_translate(f"Error embedding question: {e}", [])

        # Step 1: Search for relevant chunks
        try:
            chunks = self.db.search_similar_chunks(query_embedding, top_k=self.top_k, threshold=0.15)
        except Exception as e:
            logging.error(f"DB search error: {e}")
            return self._wrap_and_translate(f"Error searching database: {e}", [])

        if not chunks:
            return self._wrap_and_translate("Sorry, I couldn't find any relevant information in the audio.", [])

        pdf_sources = list({f"{c['metadata'].get('pdf_name','unknown')} (page {c['metadata'].get('page_num','?')})"
                            for c in chunks})

        context_parts = [
            f"[Chunk {i+1} — {c['metadata'].get('pdf_name','unknown')}, page {c['metadata'].get('page_num','?')}]\n{c.get('text','').strip()}"
            for i, c in enumerate(chunks)
        ]
        context_text = "[START TRANSCRIPT EXCERPTS]\n" + "\n\n".join(context_parts) + "\n[END TRANSCRIPT EXCERPTS]"

        # Step 2: Extraction call (deterministic)
        try:
            extraction_response = openai.chat.completions.create(
                model=self.model_extraction,
                messages=[
                    {"role": "system", "content": self.extraction_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}"}
                ],
                temperature=0.0,
                max_tokens=1200
            )
            raw_content = extraction_response.choices[0].message.content.strip()
            extracted_facts = extract_first_json(raw_content)

            if not extracted_facts:
                logging.warning("First extraction failed to produce JSON. Attempting secondary extraction.")
                extraction_response2 = openai.chat.completions.create(
                    model=self.model_extraction,
                    messages=[
                        {"role": "system", "content": self.extraction_prompt},
                        {"role": "user", "content": f"Context:\n{context_text}\n\nIMPORTANT: Output valid JSON ONLY, no commentary."}
                    ],
                    temperature=0.0,
                    max_tokens=1200
                )
                raw_content2 = extraction_response2.choices[0].message.content.strip()
                extracted_facts = extract_first_json(raw_content2)

            if not extracted_facts:
                logging.error("Extraction returned invalid JSON.")
                return self._wrap_and_translate("Extraction step failed: invalid JSON format.", pdf_sources)

        except Exception as e:
            logging.error(f"OpenAI extraction error: {e}")
            return self._wrap_and_translate(f"Error extracting facts: {e}", pdf_sources)

        # Step 3: Deterministic answer shortcuts for known intents
        try:
            lower_q = user_query.strip().lower()

            def answer_text(s: str) -> Dict:
                return self._wrap_and_translate(s.strip(), pdf_sources)

            # Helper: find participant by role
            def find_participant_by_role(role: str) -> str:
                for p in extracted_facts.get("participants", []) or []:
                    if (p.get("role") or "").lower() == role:
                        return p.get("name") or "unknown"
                cr = extracted_facts.get("call_roles", {}) or {}
                if role == "agent":
                    return cr.get("callee") or "unknown"
                if role == "customer":
                    return cr.get("caller") or "unknown"
                return "unknown"

            # WHO CALLED
            if any(kw in lower_q for kw in ["who made this call", "who called", "who initiated the call", "caller name"]):
                caller = (extracted_facts.get("call_roles", {}) or {}).get("caller") or find_participant_by_role("agent")
                if caller and caller != "unknown":
                    return answer_text(f"{caller} made the call.")

            # WHO WAS THE CUSTOMER
            if any(kw in lower_q for kw in ["who was the customer", "customer name", "who was the client", "client name"]):
                customer = find_participant_by_role("customer")
                if customer and customer != "unknown":
                    return answer_text(f"The customer on this call was {customer}.")

            # WHAT QUESTIONS <ROLE> ASKED
            if "what questions" in lower_q and "asked" in lower_q:
                target_role = None
                if any(a in lower_q for a in ["agent", "you", "representative"]):
                    target_role = "agent"
                elif any(a in lower_q for a in ["customer", "client", "caller"]):
                    target_role = "customer"

                qa_list = []
                for item in extracted_facts.get("qa_pairs", []) or []:
                    asked_by = (item.get("asked_by") or "").lower()
                    q_text = (item.get("question") or "").strip()
                    if not q_text:
                        continue
                    if target_role and asked_by == target_role:
                        qa_list.append(q_text)
                if qa_list:
                    return answer_text("Here are the questions that were asked:\n" + "\n".join(f"- {q}" for q in qa_list[:10]))

            # WHAT WAS THE CALL ABOUT
            if any(kw in lower_q for kw in ["what was the call about", "what is the call about", "summarize the call", "summary of the call"]):
                summary = extracted_facts.get("summary") or ""
                if summary:
                    return answer_text(summary)

            # FULL CALL EVALUATION shortcut
            if any(kw in lower_q for kw in ["evaluate this call", "analyze call", "call analysis", "rate the agent", "evaluate agent", "call evaluation"]):
                performance = extracted_facts.get("agent_performance", {}) or {}
                quality = extracted_facts.get("overall_call_quality", "unknown")
                summary = extracted_facts.get("summary", "")
                lines = []
                lines.append(f"Overall Call Quality: {quality}")
                if performance:
                    for k, v in performance.items():
                        lines.append(f"- {k.replace('_',' ').title()}: {v}")
                if summary:
                    lines.append(f"\nSummary: {summary}")
                # Short suggestions inferred from low scores
                try:
                    suggestions = []
                    for metric, score_reason in performance.items():
                        m = re.match(r"^\s*(\d+)", str(score_reason))
                        if m:
                            score = int(m.group(1))
                            if score <= 5:
                                suggestions.append(f"Improve {metric.replace('_',' ')} (score {score}) — consider coaching or script changes.")
                    if suggestions:
                        lines.append("\nSuggestions:")
                        for s in suggestions:
                            lines.append(f"- {s}")
                except Exception:
                    pass
                return answer_text("\n".join(lines))

        except Exception as e:
            logging.exception(f"Deterministic handling error: {e}")

        # Step 4: Final user-facing rewrite
        try:
            final_response = openai.chat.completions.create(
                model=self.model_final,
                messages=[
                    {"role": "system", "content": self.final_prompt},
                    {"role": "user", "content": f"Question: {user_query}\n\nJSON: {json.dumps(extracted_facts)}"}
                ],
                temperature=0.4,
                max_tokens=800
            )
            final_answer = final_response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"OpenAI final response error: {e}")
            final_answer = f"OpenAI API error: {e}"

        try:
            self._log_interaction(user_query, context_text, extracted_facts, final_answer)
        except Exception as e:
            logging.error(f"Logging failed: {e}")

        return self._wrap_and_translate(final_answer, pdf_sources)

    def _log_interaction(self, query: str, context: str, extracted_json: dict, response: str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(self.log_dir, f"chat_{timestamp}.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"TIMESTAMP: {timestamp}\n\nQUESTION:\n{query}\n\n")
                f.write(f"RETRIEVED CONTEXT:\n{context}\n\n")
                f.write(f"EXTRACTED JSON:\n{json.dumps(extracted_json, indent=2, ensure_ascii=False)}\n\n")
                f.write(f"RESPONSE:\n{response}\n")
            logging.info(f"Chat interaction logged to {log_path}")
        except Exception as e:
            logging.error(f"Failed to write chat log: {e}")

    def get_database_info(self) -> Dict:
        try:
            pdf_names = self.db.get_all_pdf_names()
        except Exception:
            pdf_names = []
        return {
            "total_pdfs": len(pdf_names),
            "total_chunks": self.db.get_total_chunks(),
            "pdf_names": pdf_names,
        }

    def clear_database(self):
        self.db.clear()

    def clear_conversation_history(self):
        pass
