"""
Streamlit UI for Restaurant Review Insights Bot

Usage:
  streamlit run streamlit_app.py
"""

import os
import pathlib
import sys
import streamlit as st
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from VectordB.ChromaChat2 import (
    build_prompt,
    parse_sources,
    retrieve_relevant_chunks,
    openai_client,
    pinecone_index,
)

load_dotenv()

PINECONE_INDEX = os.getenv("PINECONE_INDEX", "restaurant-bots")
MAX_RESPONSE_TOKENS = 500
FALLBACK_TEXT = "No relevant restaurant reviews were found for that question."

RESTAURANT_MAP = {
    "Crimson Coward (Burgers)": "crimson_coward",
    "Vocelli Pizza": "vocelli_pizza",
}

st.set_page_config(
    page_title="🍗 Customer Review Insights Bot",
    page_icon="🍗",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🍗 Customer Review Insights Bot")
st.write("Ask questions about restaurant reviews for Crimson Coward or Vocelli Pizza.")

selected_restaurant = st.sidebar.selectbox(
    "Choose a restaurant",
    options=list(RESTAURANT_MAP.keys()),
    index=0,
)
selected_namespace = RESTAURANT_MAP[selected_restaurant]

st.sidebar.success(f"Using Pinecone index: {PINECONE_INDEX}")
st.sidebar.success(f"Active namespace: {selected_namespace}")

with st.sidebar.expander("Index stats", expanded=False):
    try:
        stats = pinecone_index.describe_index_stats(namespace=selected_namespace)
        total = stats.get("namespaces", {}).get(selected_namespace, {}).get("vector_count") if isinstance(stats, dict) else None
        if total is None:
            total = getattr(stats, "total_vector_count", None)
        if total is not None:
            st.write(f"Total vectors in namespace: {total:,}")
        else:
            st.write("Index stats unavailable.")
    except Exception as e:
        st.write(f"Could not load index stats: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("Use the restaurant selector above to choose the review namespace.")

with st.sidebar.expander("Advanced settings", expanded=False):
    top_k = st.slider("Retrieved chunks (top_k)", min_value=3, max_value=50, value=12, step=1)
    show_debug = st.checkbox("Show debug output", value=False)

prompt = st.chat_input("Ask a restaurant review question...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving review context and generating an answer..."):
            embedded_chunks = retrieve_relevant_chunks(prompt, NAMESPACE=selected_namespace, top_k=top_k)

            if not embedded_chunks:
                st.info(FALLBACK_TEXT)
            else:
                prompt_text = build_prompt(prompt, embedded_chunks)

                response = None
                for attempt in range(3):
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": prompt_text}],
                            max_tokens=MAX_RESPONSE_TOKENS,
                            temperature=0.3,
                        )
                        break
                    except Exception as e:
                        if attempt >= 2:
                            st.error(f"OpenAI error: {e}")
                        else:
                            continue

                if not response:
                    st.error("Sorry, I couldn't get a response. Please try again.")
                else:
                    full_answer = response.choices[0].message.content.strip()
                    ans_text, sources = parse_sources(full_answer)
                    st.markdown(ans_text)
                    if sources:
                        st.markdown("\n📚 **Sources:**")
                        for title, url in sources:
                            st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown("\n📚 **Sources:** None cited.")

                    if show_debug:
                        st.expander("Debug", expanded=False).write({
                            "prompt_text": prompt_text,
                            "embedded_chunks": embedded_chunks[:top_k],
                        })
