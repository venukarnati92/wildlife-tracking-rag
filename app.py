"""Streamlit chat UI for Wildlife Tracking RAG."""

from __future__ import annotations

import os

import streamlit as st

from assistant import create_assistant
from db_feedback import save_feedback
from db_save import save_conversation
from judge import evaluate_relevance


st.set_page_config(page_title="Wildlife Tracking RAG", page_icon=":paw_prints:", layout="wide")
st.title("Wildlife Tracking RAG")
st.caption(
    "Ask questions about wildlife tracking studies. Answers are grounded in "
    "Movebank study metadata (species, investigators, locations, deployments, citations)."
)


STRATEGIES = {
    "Hybrid + rerank + query rewrite (best)": "hybrid_rerank_rewrite",
    "Hybrid + rerank": "hybrid_rerank",
    "Hybrid (text + vector)": "hybrid",
    "Vector only": "vector",
    "Text only (minsearch)": "text",
}


@st.cache_resource(show_spinner="Loading index and models...")
def get_assistant(strategy_key: str):
    return create_assistant(strategy=STRATEGIES[strategy_key])


DEFAULT_STRATEGY_LABEL = "Hybrid (text + vector)"

with st.sidebar:
    st.header("Retrieval strategy")
    strategy_options = list(STRATEGIES.keys())
    strategy_label = st.selectbox(
        "Choose search strategy",
        strategy_options,
        index=strategy_options.index(DEFAULT_STRATEGY_LABEL),
    )
    st.caption(f"Backend model: {os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    st.caption("Postgres logs every conversation for the dashboard.")


assistant = get_assistant(strategy_label)

user_input = st.text_input("Your question", placeholder="Which studies track turkey vultures?")

if st.button("Ask", type="primary") and user_input:
    with st.spinner("Retrieving and generating..."):
        answer = assistant.rag(user_input)

    record = assistant.last_call
    st.markdown("### Answer")
    st.write(answer)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latency", f"{record.response_time:.2f}s")
    col2.metric("Prompt tokens", record.prompt_tokens)
    col3.metric("Completion tokens", record.completion_tokens)
    col4.metric("Cost", f"${record.cost:.6f}")

    with st.expander("Retrieved study context"):
        for doc in assistant.last_search_results or []:
            st.markdown(f"**{doc.get('name')}** — study `{doc.get('study_id')}`")
            st.markdown(f"- Species: {doc.get('taxa') or 'n/a'}")
            st.markdown(f"- PI: {doc.get('principal_investigator') or 'n/a'}")
            st.markdown(f"- Period: {doc.get('time_period') or 'n/a'}")
            st.divider()

    retrieved_ids = [d.get("study_id") for d in (assistant.last_search_results or []) if d.get("study_id")]

    try:
        conversation_id = save_conversation(
            record, user_input, STRATEGIES[strategy_label], retrieved_ids=retrieved_ids
        )
        st.session_state["conversation_id"] = conversation_id

        relevance, explanation = evaluate_relevance(user_input, answer)
        save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)
        st.caption(f"Judge: **{relevance}** — {explanation}")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not persist conversation ({exc}). Is Postgres running?")


if "conversation_id" in st.session_state:
    col1, col2, _ = st.columns([1, 1, 6])
    with col1:
        if st.button("\U0001F44D", help="Good answer"):
            save_feedback(st.session_state["conversation_id"], "user", score=1)
            st.toast("Thanks!")
    with col2:
        if st.button("\U0001F44E", help="Bad answer"):
            save_feedback(st.session_state["conversation_id"], "user", score=-1)
            st.toast("Thanks for the feedback!")

st.markdown("---")
st.markdown("**Sample questions**")
for sample in [
    "Which studies track turkey vultures?",
    "Are there any studies on African elephant movement?",
    "What sensors are used to track Galapagos albatrosses?",
    "Which studies span more than 5 years of data?",
    "Who are the principal investigators for GPS-tracked seabird studies?",
]:
    st.markdown(f"- {sample}")
