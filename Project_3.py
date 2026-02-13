from __future__ import annotations
import json
import re
import time
import uuid
import sqlite3
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

# --- Optional sentiment (VADER). If not available, fallback to lexicon. ---
USE_VADER = False
try:
    import nltk  # type: ignore
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        nltk.download("vader_lexicon", quiet=True)
    _vader = SentimentIntensityAnalyzer()
    USE_VADER = True
except Exception:
    _vader = None
    USE_VADER = False

# --- ML-lite for intent + FAQ retrieval ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


# =========================
# Config / Data (edit here)
# =========================

DEFAULT_TOP_K_FAQ = 3
INTENT_THRESHOLD = 0.55  # if below => fallback/clarify
FAQ_SIM_THRESHOLD = 0.33  # if below => not confident in KB

# Example intents (you can expand)
INTENTS_TRAINING = [
    ("greet", "hi"),
    ("greet", "hello"),
    ("greet", "hey"),
    ("goodbye", "bye"),
    ("goodbye", "see you later"),
    ("thanks", "thanks"),
    ("thanks", "thank you"),
    ("help", "what can you do"),
    ("help", "help me"),
    ("faq", "what is your refund policy"),
    ("faq", "how do i reset my password"),
    ("support_ticket", "create a support ticket"),
    ("support_ticket", "i want to raise a complaint"),
    ("order_status", "track my order"),
    ("order_status", "where is my order"),
    ("handoff", "talk to a human"),
    ("handoff", "connect me to agent"),
]

# Simple Knowledge Base / FAQs
FAQ_KB = [
    {
        "id": "faq_refund",
        "q": "What is your refund policy?",
        "a": "Refunds are processed within 5–7 business days after approval. If you paid by card, it may take extra 2–3 days to reflect in your account.",
        "tags": ["refund", "policy", "payment"],
    },
    {
        "id": "faq_password_reset",
        "q": "How do I reset my password?",
        "a": "Go to Login → 'Forgot Password' → enter your email → verify OTP → set a new password. If OTP doesn't arrive, check spam or try again after 2 minutes.",
        "tags": ["password", "reset", "otp", "login"],
    },
    {
        "id": "faq_support_hours",
        "q": "What are your support hours?",
        "a": "Support is available Mon–Sat, 9:00 AM–7:00 PM IST. For urgent issues, you can request a human handoff.",
        "tags": ["support", "hours"],
    },
]


# =========================
# Utility: SQLite storage
# =========================

DB_PATH = "chatbot.sqlite3"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            channel TEXT,
            created_at REAL,
            updated_at REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT, -- 'user' or 'assistant'
            text TEXT,
            ts REAL,
            intent TEXT,
            intent_conf REAL,
            sentiment TEXT,
            sentiment_score REAL,
            entities_json TEXT,
            latency_ms INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            message_id TEXT,
            rating INTEGER, -- -1, 0, +1
            comment TEXT,
            ts REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_log (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            tool_name TEXT,
            input_json TEXT,
            output_json TEXT,
            latency_ms INTEGER,
            ts REAL
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_session(session_id: str, user_id: str, channel: str = "api") -> None:
    conn = db()
    cur = conn.cursor()
    now = time.time()
    cur.execute(
        """
        INSERT INTO sessions(session_id, user_id, channel, created_at, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET
            user_id=excluded.user_id,
            channel=excluded.channel,
            updated_at=excluded.updated_at
        """,
        (session_id, user_id, channel, now, now),
    )
    conn.commit()
    conn.close()


def store_message(
    session_id: str,
    role: str,
    text: str,
    intent: str,
    intent_conf: float,
    sentiment: str,
    sentiment_score: float,
    entities: Dict[str, Any],
    latency_ms: int,
) -> str:
    conn = db()
    cur = conn.cursor()
    mid = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO messages(id, session_id, role, text, ts, intent, intent_conf,
                             sentiment, sentiment_score, entities_json, latency_ms)
        VALUES(?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            mid,
            session_id,
            role,
            text,
            time.time(),
            intent,
            float(intent_conf),
            sentiment,
            float(sentiment_score),
            json.dumps(entities, ensure_ascii=False),
            int(latency_ms),
        ),
    )
    conn.commit()
    conn.close()
    return mid


def store_tool_log(
    session_id: str, tool_name: str, inp: Dict[str, Any], out: Dict[str, Any], latency_ms: int
) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tool_log(id, session_id, tool_name, input_json, output_json, latency_ms, ts)
        VALUES(?,?,?,?,?,?,?)
        """,
        (
            str(uuid.uuid4()),
            session_id,
            tool_name,
            json.dumps(inp, ensure_ascii=False),
            json.dumps(out, ensure_ascii=False),
            int(latency_ms),
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


# =========================
# NLP: Sentiment, NER, Intent
# =========================

POS_WORDS = {"good", "great", "awesome", "nice", "love", "perfect", "excellent", "thanks", "thank"}
NEG_WORDS = {"bad", "worst", "hate", "angry", "refund", "issue", "problem", "terrible", "frustrated", "not working"}


def analyze_sentiment(text: str) -> Tuple[str, float]:
    t = text.strip()
    if not t:
        return "neutral", 0.0

    if USE_VADER and _vader is not None:
        score = _vader.polarity_scores(t)["compound"]
        if score >= 0.2:
            return "positive", float(score)
        if score <= -0.2:
            return "negative", float(score)
        return "neutral", float(score)

    # fallback lexicon
    lowered = t.lower()
    p = sum(1 for w in POS_WORDS if w in lowered)
    n = sum(1 for w in NEG_WORDS if w in lowered)
    score = (p - n) / max(1, (p + n))
    if score > 0.2:
        return "positive", float(score)
    if score < -0.2:
        return "negative", float(score)
    return "neutral", float(score)


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-\s]?)?(?:[6-9]\d{9}|\d{3}[-\s]?\d{3}[-\s]?\d{4})\b")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
ORDER_RE = re.compile(r"\b(?:order|ord|tracking)\s*#?\s*([A-Za-z0-9-]{5,})\b", re.IGNORECASE)


def extract_entities(text: str) -> Dict[str, Any]:
    entities: Dict[str, Any] = {}
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    dates = DATE_RE.findall(text)

    order_match = ORDER_RE.search(text)
    order_id = order_match.group(1) if order_match else None

    if emails:
        entities["emails"] = emails
    if phones:
        entities["phones"] = phones
    if dates:
        entities["dates"] = dates
    if order_id:
        entities["order_id"] = order_id

    # naive PERSON extraction (optional): "my name is X"
    m = re.search(r"\bmy name is\s+([A-Za-z][A-Za-z ]{1,40})\b", text, re.IGNORECASE)
    if m:
        entities["person_name"] = m.group(1).strip()

    return entities


@dataclass
class IntentResult:
    label: str
    confidence: float
    topk: List[Tuple[str, float]]


class IntentModel:
    def __init__(self, training_pairs: List[Tuple[str, str]]):
        labels = [y for y, _ in training_pairs]
        texts = [x for _, x in training_pairs]

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        X = self.vectorizer.fit_transform(texts)

        self.clf = LogisticRegression(max_iter=400)
        self.clf.fit(X, labels)

        self.label_list = sorted(set(labels))

    def predict(self, text: str, top_k: int = 3) -> IntentResult:
        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        classes = self.clf.classes_

        pairs = sorted([(classes[i], float(probs[i])) for i in range(len(classes))], key=lambda x: x[1], reverse=True)
        best_label, best_conf = pairs[0]
        return IntentResult(best_label, best_conf, pairs[:top_k])


# =========================
# KB Retrieval (Simple RAG)
# =========================

class FAQRetriever:
    def __init__(self, faq_items: List[Dict[str, Any]]):
        self.faq_items = faq_items
        corpus = []
        for item in faq_items:
            corpus.append(f"{item['q']} {' '.join(item.get('tags', []))}")
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K_FAQ) -> List[Tuple[Dict[str, Any], float]]:
        qv = self.vectorizer.transform([query])
        sims = (self.matrix @ qv.T).toarray().ravel()
        idx = np.argsort(-sims)[:top_k]
        return [(self.faq_items[i], float(sims[i])) for i in idx]


# =========================
# Memory (short + long)
# =========================

class MemoryStore:
    """
    Short-term: in-memory dict
    Long-term: SQLite messages already stored; we also keep lightweight summaries in memory for speed
    """

    def __init__(self):
        self.short_turns: Dict[str, List[Dict[str, Any]]] = {}   # session_id -> list of turns
        self.session_summary: Dict[str, str] = {}                # session_id -> summary

    def add_turn(self, session_id: str, role: str, text: str, entities: Dict[str, Any]) -> None:
        self.short_turns.setdefault(session_id, []).append(
            {"role": role, "text": text, "entities": entities, "ts": time.time()}
        )
        # keep only last N turns
        if len(self.short_turns[session_id]) > 12:
            self.short_turns[session_id] = self.short_turns[session_id][-12:]

    def get_recent(self, session_id: str, n: int = 6) -> List[Dict[str, Any]]:
        turns = self.short_turns.get(session_id, [])
        return turns[-n:]

    def update_summary(self, session_id: str) -> None:
        turns = self.short_turns.get(session_id, [])
        if not turns:
            return
        # Simple summarizer heuristic (no LLM): pick last user problem + key entities
        user_msgs = [t["text"] for t in turns if t["role"] == "user"]
        last_user = user_msgs[-1] if user_msgs else ""
        ents = {}
        for t in turns:
            ents.update(t.get("entities", {}))
        pieces = []
        if last_user:
            pieces.append(f"Last user query: {last_user[:140]}")
        if ents:
            pieces.append(f"Known entities: {json.dumps(ents, ensure_ascii=False)}")
        self.session_summary[session_id] = " | ".join(pieces)[:600]

    def get_summary(self, session_id: str) -> str:
        return self.session_summary.get(session_id, "")


# =========================
# Orchestrator (Brain)
# =========================

class Orchestrator:
    def __init__(self):
        self.intent_model = IntentModel(INTENTS_TRAINING)
        self.retriever = FAQRetriever(FAQ_KB)
        self.memory = MemoryStore()

        # simple in-memory analytics
        self.metrics = {
            "messages_total": 0,
            "fallback_total": 0,
            "handoff_total": 0,
            "intents": {},  # label -> count
            "avg_latency_ms": 0.0,
            "latency_samples": 0,
        }

    def _update_metrics(self, intent: str, latency_ms: int, fallback: bool = False, handoff: bool = False) -> None:
        self.metrics["messages_total"] += 1
        self.metrics["intents"][intent] = self.metrics["intents"].get(intent, 0) + 1
        if fallback:
            self.metrics["fallback_total"] += 1
        if handoff:
            self.metrics["handoff_total"] += 1

        self.metrics["latency_samples"] += 1
        n = self.metrics["latency_samples"]
        prev = self.metrics["avg_latency_ms"]
        self.metrics["avg_latency_ms"] = prev + (latency_ms - prev) / n

    def handle_user_message(self, session_id: str, user_id: str, text: str) -> Dict[str, Any]:
        t0 = time.time()
        upsert_session(session_id, user_id, channel="api")

        # NLP
        entities = extract_entities(text)
        sentiment, sent_score = analyze_sentiment(text)
        intent_res = self.intent_model.predict(text)

        # Memory update
        self.memory.add_turn(session_id, "user", text, entities)
        self.memory.update_summary(session_id)

        # Routing
        reply_text, route_info, used_fallback, used_handoff = self.route_and_respond(
            session_id=session_id,
            user_id=user_id,
            user_text=text,
            intent=intent_res,
            entities=entities,
            sentiment=sentiment,
        )

        latency_ms = int((time.time() - t0) * 1000)

        # Store user + assistant messages
        store_message(
            session_id=session_id,
            role="user",
            text=text,
            intent=intent_res.label,
            intent_conf=intent_res.confidence,
            sentiment=sentiment,
            sentiment_score=sent_score,
            entities=entities,
            latency_ms=latency_ms,
        )
        self.memory.add_turn(session_id, "assistant", reply_text, {})

        store_message(
            session_id=session_id,
            role="assistant",
            text=reply_text,
            intent=intent_res.label,
            intent_conf=intent_res.confidence,
            sentiment="neutral",
            sentiment_score=0.0,
            entities={},
            latency_ms=latency_ms,
        )

        self._update_metrics(intent_res.label, latency_ms, fallback=used_fallback, handoff=used_handoff)

        return {
            "session_id": session_id,
            "user_id": user_id,
            "intent": {
                "label": intent_res.label,
                "confidence": intent_res.confidence,
                "topk": intent_res.topk,
            },
            "sentiment": {"label": sentiment, "score": sent_score},
            "entities": entities,
            "reply": reply_text,
            "route": route_info,
            "latency_ms": latency_ms,
            "summary": self.memory.get_summary(session_id),
        }

    def route_and_respond(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        intent: IntentResult,
        entities: Dict[str, Any],
        sentiment: str,
    ) -> Tuple[str, Dict[str, Any], bool, bool]:
        # Sentiment adaptation
        empathy = ""
        if sentiment == "negative":
            empathy = "I’m sorry about that. "
        elif sentiment == "positive":
            empathy = "Nice! "

        # Low confidence -> clarify/fallback
        if intent.confidence < INTENT_THRESHOLD:
            msg = (
                empathy
                + "I want to make sure I understand. Are you asking about (1) password reset, (2) refunds, (3) order tracking, or (4) support ticket?"
            )
            return msg, {"strategy": "fallback_low_intent_conf"}, True, False

        label = intent.label

        # Explicit handoff
        if label == "handoff":
            return (
                empathy
                + "Sure — I can connect you to a human agent. Please share your issue in 1–2 lines and your preferred contact (email/phone), and I’ll prepare a ticket for handoff."
            ), {"strategy": "handoff"}, False, True

        # Greetings/thanks
        if label == "greet":
            return empathy + "Hi! Tell me what you need help with (refund / password reset / order tracking / support ticket).", {"strategy": "template"}, False, False
        if label == "thanks":
            return empathy + "You’re welcome! If you need anything else, just ask.", {"strategy": "template"}, False, False
        if label == "goodbye":
            return "Bye! Take care.", {"strategy": "template"}, False, False
        if label == "help":
            return (
                "I can help with:\n"
                "• FAQs (refund policy, password reset, support hours)\n"
                "• Order tracking (if you share an order/tracking ID)\n"
                "• Creating a support ticket\n"
                "• Connecting you to a human agent\n"
                "What do you want to do?"
            ), {"strategy": "help_menu"}, False, False

        # FAQ / KB retrieval (simple RAG)
        if label == "faq":
            t0 = time.time()
            hits = self.retriever.retrieve(user_text, top_k=DEFAULT_TOP_K_FAQ)
            tool_latency = int((time.time() - t0) * 1000)

            best_item, best_score = hits[0]
            store_tool_log(
                session_id,
                "faq_retriever_tfidf",
                {"query": user_text, "top_k": DEFAULT_TOP_K_FAQ},
                {"best_id": best_item["id"], "best_score": best_score, "hits": [(h[0]["id"], h[1]) for h in hits]},
                tool_latency,
            )

            if best_score < FAQ_SIM_THRESHOLD:
                msg = empathy + "I’m not fully sure which FAQ you mean. Is it about refunds, password reset, or support hours?"
                return msg, {"strategy": "faq_low_similarity", "best_score": best_score}, True, False

            # Return the best answer (grounded)
            return (
                empathy + best_item["a"]
                + "\n\nIf you want, tell me your exact situation and I’ll guide step-by-step."
            ), {"strategy": "faq_retrieval", "kb_id": best_item["id"], "score": best_score}, False, False

        # Simulated order status "tool"
        if label == "order_status":
            order_id = entities.get("order_id")
            if not order_id:
                return (
                    empathy
                    + "I can help track it. Please share your Order ID / Tracking ID (example: Order #A1B2C3)."
                ), {"strategy": "order_need_id"}, True, False

            # Fake status (replace with real API call)
            t0 = time.time()
            status = {
                "order_id": order_id,
                "status": "In Transit",
                "eta": "2–4 days",
                "last_update": "Package left sorting facility",
            }
            tool_latency = int((time.time() - t0) * 1000)
            store_tool_log(session_id, "order_status_mock", {"order_id": order_id}, status, tool_latency)

            return (
                empathy
                + f"Order **{order_id}** status: **{status['status']}**\n"
                + f"ETA: {status['eta']}\n"
                + f"Latest: {status['last_update']}"
            ), {"strategy": "order_status_tool", "order_id": order_id}, False, False

        # Simulated support ticket "tool"
        if label == "support_ticket":
            # Need contact info ideally
            email = (entities.get("emails") or [None])[0]
            phone = (entities.get("phones") or [None])[0]
            if not (email or phone):
                return (
                    empathy
                    + "I can create a support ticket. Please share your email or phone, and describe the issue in 1–2 lines."
                ), {"strategy": "ticket_need_contact"}, True, False

            t0 = time.time()
            ticket_id = "TCK-" + str(uuid.uuid4())[:8].upper()
            out = {"ticket_id": ticket_id, "contact": email or phone, "status": "created"}
            tool_latency = int((time.time() - t0) * 1000)
            store_tool_log(session_id, "support_ticket_mock", {"contact": email or phone, "issue": user_text}, out, tool_latency)

            return (
                empathy
                + f"Done ✅ Your ticket **{ticket_id}** has been created. A support agent will contact you at **{email or phone}**."
            ), {"strategy": "ticket_tool", "ticket_id": ticket_id}, False, False

        # Default
        msg = empathy + "I can help with FAQs, order tracking, and support tickets. What exactly do you want to do?"
        return msg, {"strategy": "default_fallback"}, True, False


# =========================
# FastAPI App + Schemas
# =========================

app = FastAPI(title="Dynamic AI Chatbot (Single File)", version="1.0")

init_db()
BOT = Orchestrator()


class ChatRequest(BaseModel):
    user_id: str = Field(..., examples=["u1"])
    session_id: str = Field(default_factory=lambda: "s-" + str(uuid.uuid4())[:8])
    text: str = Field(..., examples=["Hi, I need to reset my password"])
    channel: str = Field(default="api", examples=["api", "web", "telegram"])


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    reply: str
    intent: Dict[str, Any]
    sentiment: Dict[str, Any]
    entities: Dict[str, Any]
    route: Dict[str, Any]
    latency_ms: int
    summary: str


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(..., ge=-1, le=1)  # -1, 0, +1
    comment: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "vader": USE_VADER, "db": DB_PATH}


@app.post("/chat/send", response_model=ChatResponse)
def chat_send(req: ChatRequest) -> Dict[str, Any]:
    upsert_session(req.session_id, req.user_id, channel=req.channel)
    out = BOT.handle_user_message(req.session_id, req.user_id, req.text)
    return out


@app.websocket("/chat/ws")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        session_id = ws.query_params.get("session_id") or ("s-" + str(uuid.uuid4())[:8])
        user_id = ws.query_params.get("user_id") or "anonymous"
        upsert_session(session_id, user_id, channel="websocket")

        await ws.send_json({"type": "system", "message": "Connected", "session_id": session_id, "user_id": user_id})

        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                text = str(payload.get("text", "")).strip()
            except Exception:
                text = data.strip()

            if not text:
                await ws.send_json({"type": "error", "message": "Empty message"})
                continue

            out = BOT.handle_user_message(session_id, user_id, text)
            await ws.send_json({"type": "assistant", "data": out})

    except WebSocketDisconnect:
        return
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    fid = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO feedback(id, message_id, rating, comment, ts) VALUES(?,?,?,?,?)",
        (fid, req.message_id, int(req.rating), req.comment, time.time()),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "feedback_id": fid}


@app.get("/analytics/summary")
def analytics_summary() -> Dict[str, Any]:
    # combine in-memory + DB counts (DB is source of truth)
    conn = db()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM messages")
    messages = int(cur.fetchone()["c"])
    cur.execute("SELECT COUNT(*) AS c FROM sessions")
    sessions = int(cur.fetchone()["c"])
    cur.execute("SELECT COUNT(*) AS c FROM feedback")
    feedback = int(cur.fetchone()["c"])

    cur.execute(
        """
        SELECT intent, COUNT(*) AS c
        FROM messages
        WHERE role='user'
        GROUP BY intent
        ORDER BY c DESC
        LIMIT 20
        """
    )
    intents = [{"intent": r["intent"], "count": int(r["c"])} for r in cur.fetchall()]

    cur.execute(
        """
        SELECT AVG(latency_ms) AS avg_lat, MAX(latency_ms) AS max_lat
        FROM messages
        """
    )
    row = cur.fetchone()
    avg_lat = float(row["avg_lat"] or 0.0)
    max_lat = int(row["max_lat"] or 0)

    conn.close()

    fallback_rate = (
        (BOT.metrics["fallback_total"] / max(1, BOT.metrics["messages_total"])) * 100.0
    )

    return {
        "db": {"sessions": sessions, "messages": messages, "feedback": feedback},
        "runtime": {
            "messages_total": BOT.metrics["messages_total"],
            "fallback_total": BOT.metrics["fallback_total"],
            "handoff_total": BOT.metrics["handoff_total"],
            "fallback_rate_percent": round(fallback_rate, 2),
            "avg_latency_ms_runtime": round(BOT.metrics["avg_latency_ms"], 2),
        },
        "latency_db": {"avg_latency_ms": round(avg_lat, 2), "max_latency_ms": max_lat},
        "top_intents_db": intents,
    }


@app.get("/analytics/conversations")
def analytics_conversations(limit: int = 50) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT session_id, user_id, channel, created_at, updated_at
        FROM sessions
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (int(limit),),
    )
    sessions = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"sessions": sessions}


@app.get("/chat/history")
def chat_history(session_id: str, limit: int = 50) -> Dict[str, Any]:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, role, text, ts, intent, intent_conf, sentiment, sentiment_score, entities_json, latency_ms
        FROM messages
        WHERE session_id=?
        ORDER BY ts DESC
        LIMIT ?
        """,
        (session_id, int(limit)),
    )
    rows = []
    for r in cur.fetchall():
        d = dict(r)
        try:
            d["entities_json"] = json.loads(d["entities_json"] or "{}")
        except Exception:
            pass
        rows.append(d)
    conn.close()
    rows.reverse()
    return {"session_id": session_id, "messages": rows, "summary": BOT.memory.get_summary(session_id)}


# =========================
# Main
# =========================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chatbot:app", host="127.0.0.1", port=8000, reload=True)
