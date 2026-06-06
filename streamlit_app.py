"""
Streamlit UI for Customer Review Insights Bot

Usage:
  streamlit run streamlit_app.py
"""

import json
import os
import time
import uuid
import streamlit as st
from typing import Dict, Optional

from app.rag import generate_grounded_response
from app.restaurants import load_restaurant_map, validate_index_name

st.set_page_config(
    page_title="🍗 Customer Review Insights Bot",
    page_icon="🍗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Session state helpers
# =========================

def _init_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    if "settings" not in st.session_state:
        st.session_state.settings = {
            "top_k": 12,
            "min_recurring_reviews": 2,
            "debug_on": False,
        }

    if "selected_restaurant" not in st.session_state:
        st.session_state.selected_restaurant = None

    if "custom_index_name" not in st.session_state:
        st.session_state.custom_index_name = ""


def _new_chat() -> None:
    chat_id = str(uuid.uuid4())
    st.session_state.conversations[chat_id] = {
        "title": "New chat",
        "messages": [],
        "created": time.time(),
    }
    st.session_state.active_chat_id = chat_id


def _ensure_active_chat() -> None:
    if not st.session_state.conversations:
        _new_chat()
        return
    if st.session_state.active_chat_id not in st.session_state.conversations:
        newest = max(st.session_state.conversations.items(), key=lambda kv: kv[1]["created"])[0]
        st.session_state.active_chat_id = newest


def _set_title_from_first_user_message(chat_id: str) -> None:
    convo = st.session_state.conversations[chat_id]
    if convo["title"] != "New chat":
        return
    for m in convo["messages"]:
        if m["role"] == "user" and m["content"].strip():
            t = m["content"].strip()
            convo["title"] = t[:40] + ("…" if len(t) > 40 else "")
            return


def _delete_chat(chat_id: str) -> None:
    st.session_state.conversations.pop(chat_id, None)
    if not st.session_state.conversations:
        _new_chat()
    else:
        newest = max(st.session_state.conversations.items(), key=lambda kv: kv[1]["created"])[0]
        st.session_state.active_chat_id = newest


def _sorted_chat_ids_newest_first():
    items = sorted(
        st.session_state.conversations.items(),
        key=lambda kv: kv[1]["created"],
        reverse=True,
    )
    return [cid for cid, _ in items]


# =========================
# Init
# =========================
_init_state()
_ensure_active_chat()

restaurant_map = load_restaurant_map()
if not restaurant_map:
    restaurant_map = {
        "Crimson Coward": "crimson_coward",
        "Vocelli Pizza": "vocelli_pizza",
    }

available_restaurants = [
    "Crimson Coward",
    "Vocelli Pizza",
]

# Ensure we show only configured restaurants in the chooser.
available_restaurants = [name for name in available_restaurants if name in restaurant_map]
if not available_restaurants:
    available_restaurants = list(restaurant_map.keys())

if st.session_state.selected_restaurant not in available_restaurants:
    st.session_state.selected_restaurant = available_restaurants[0]

# =========================
# Sidebar — restaurant + conversation dashboard
# =========================
with st.sidebar:
    st.header("🍽️ Restaurant Selector")
    selected_rest = st.selectbox(
        "Choose restaurant",
        options=available_restaurants,
        index=available_restaurants.index(st.session_state.selected_restaurant),
        key="selected_restaurant",
    )

    selected_namespace = restaurant_map.get(selected_rest)
    ok, msg = validate_index_name(selected_namespace)
    if ok:
        st.success(f"Using Pinecone namespace: {selected_namespace}")
    else:
        st.warning(f"Selected restaurant has no configured Pinecone namespace: {msg}")

    st.markdown("---")
    st.header("💬 Conversations")
    if st.button("➕ New chat", use_container_width=True):
        _new_chat()
        st.experimental_rerun()

    st.markdown("---")
    chat_ids = _sorted_chat_ids_newest_first()
    active = st.session_state.active_chat_id
    selected = st.radio(
        "History",
        options=chat_ids,
        index=chat_ids.index(active) if active in chat_ids else 0,
        format_func=lambda cid: st.session_state.conversations[cid]["title"],
        label_visibility="collapsed",
    )
    st.session_state.active_chat_id = selected
    _ensure_active_chat()

    st.markdown("---")
    convo = st.session_state.conversations[st.session_state.active_chat_id]
    with st.expander("✏️ Rename / Manage", expanded=False):
        new_title = st.text_input("Conversation title", value=convo["title"])
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save title", use_container_width=True):
                convo["title"] = (new_title.strip() or "Untitled")
                st.experimental_rerun()
        with col2:
            if st.button("Delete chat", use_container_width=True):
                _delete_chat(st.session_state.active_chat_id)
                st.experimental_rerun()

# =========================
# Main — Chat UI
# =========================
main_pinecone_index = os.getenv("PINECONE_INDEX", "restaurant-bots")
st.title("🍗 Customer Review Insights Bot")
st.caption("Ask questions about Crimson Coward or Vocelli Pizza reviews. The bot retrieves grounded insights from your restaurant dataset.")
st.info(f"Selected restaurant: **{selected_rest}** | Pinecone index: `{main_pinecone_index}` | Namespace: `{selected_namespace}`")

cid = st.session_state.active_chat_id
convo = st.session_state.conversations[cid]

with st.expander("⚙️ Advanced settings", expanded=False):
    st.session_state.settings["top_k"] = st.slider(
        "Retrieved chunks (top_k)",
        min_value=3,
        max_value=50,
        value=int(st.session_state.settings["top_k"]),
        step=1,
    )
    st.session_state.settings["min_recurring_reviews"] = st.slider(
        "Min recurring reviews",
        min_value=1,
        max_value=10,
        value=int(st.session_state.settings["min_recurring_reviews"]),
        step=1,
    )
    st.session_state.settings["debug_on"] = st.checkbox(
        "Enable debug output",
        value=bool(st.session_state.settings["debug_on"]),
    )

for msg in convo["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a restaurant review question…")

if prompt:
    if not ok:
        st.error("Selected restaurant is not configured properly. Please choose a valid restaurant or update restaurants.json.")
    else:
        convo["messages"].append({"role": "user", "content": prompt})
        _set_title_from_first_user_message(cid)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating a grounded response for the selected restaurant…"):
                out = generate_grounded_response(
                    query=prompt,
                    top_k=int(st.session_state.settings["top_k"]),
                    min_recurring_reviews=int(st.session_state.settings["min_recurring_reviews"]),
                    include_debug=bool(st.session_state.settings["debug_on"]),
                    namespace=selected_namespace,
                )

            if not isinstance(out, dict):
                st.error("Unexpected backend output. Please check the app logs.")
                st.write(out)
            else:
                answer = out.get("answer_summary", "").strip() or "No answer summary returned."
                st.markdown(answer)

                with st.expander("Details (themes, issues, SMS, ops)", expanded=False):
                    overall = out.get("overall_sentiment") or {}
                    st.markdown(f"**Overall sentiment:** `{overall.get('label', 'mixed')}`")
                    if overall.get("rationale"):
                        st.write(overall["rationale"])

                    st.subheader("Themes")
                    themes = out.get("top_themes", []) or []
                    if not themes:
                        st.write("_No themes returned._")
                    else:
                        for th in themes:
                            st.markdown(f"**{th.get('theme','Theme')}** • sentiment: `{th.get('sentiment','mixed')}`")
                            for ev in th.get("evidence", []) or []:
                                st.markdown(f"- `{ev.get('chunk_id','')}` — {ev.get('excerpt','')}")

                st.subheader("Recurring issues")
                recurring = out.get("recurring_issues", []) or []
                if not recurring:
                    st.write("_None._")
                else:
                    for item in recurring:
                        st.markdown(f"- {item.get('issue','')}")

                st.subheader("Draft SMS")
                sms_messages = (out.get("sms_draft") or {}).get("messages", []) or []
                if not sms_messages:
                    st.write("_None._")
                else:
                    for i, msg in enumerate(sms_messages, start=1):
                        st.markdown(f"**Message {i}**")
                        st.code(msg)

                st.subheader("Source chunk IDs")
                sources = out.get("sources", []) or []
                if sources:
                    st.write("\n".join(sources))
                else:
                    st.write("_No sources returned._")

            if st.session_state.settings["debug_on"]:
                with st.expander("🧪 Debug", expanded=False):
                    st.json(out)

            convo["messages"].append({"role": "assistant", "content": answer})
