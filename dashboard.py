"""Streamlit monitoring dashboard (>=5 charts) for Wildlife Tracking RAG."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from db_query import (
    get_conversations,
    get_conversations_per_day,
    get_relevance_stats,
    get_stats,
    get_strategy_stats,
    get_top_questions,
    get_user_feedback_stats,
)


st.set_page_config(page_title="Wildlife RAG Dashboard", page_icon=":bar_chart:", layout="wide")
st.title("Wildlife Tracking RAG - Monitoring")

stats = get_stats()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total conversations", stats.total)
c2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
c3.metric("Total cost", f"${stats.total_cost:.4f}")
c4.metric("Avg tokens / call", f"{stats.avg_tokens:.0f}")

st.divider()

conversations = get_conversations(limit=500)
df = pd.DataFrame(conversations)

if df.empty:
    st.info("No conversations yet. Ask a question in the app or run `python generate_data.py`.")
    st.stop()


st.subheader("1) Conversations over time")
per_hour = pd.DataFrame(get_conversations_per_day())
if not per_hour.empty:
    per_hour = per_hour.rename(columns={"bucket": "time", "count": "conversations"})
    st.line_chart(per_hour, x="time", y="conversations")

st.subheader("2) Cost per call over time")
st.line_chart(df.sort_values("timestamp"), x="timestamp", y="cost")

st.subheader("3) Response time over time")
st.line_chart(df.sort_values("timestamp"), x="timestamp", y="response_time")

st.subheader("4) Judge relevance distribution")
relevance = get_relevance_stats()
if relevance:
    st.bar_chart(pd.DataFrame({"count": relevance}))
else:
    st.caption("No judge feedback yet.")

st.subheader("5) User feedback (thumbs)")
thumbs_up, thumbs_down = get_user_feedback_stats()
fc1, fc2 = st.columns(2)
fc1.metric(":thumbsup:", thumbs_up)
fc2.metric(":thumbsdown:", thumbs_down)

st.subheader("6) Retrieval strategy usage & latency")
strat = pd.DataFrame(get_strategy_stats())
if not strat.empty:
    st.dataframe(strat, use_container_width=True)
    st.bar_chart(strat.set_index("strategy")[["count"]])
    st.bar_chart(strat.set_index("strategy")[["avg_latency"]])

st.subheader("7) Top asked questions")
top_q = pd.DataFrame(get_top_questions(limit=10))
if not top_q.empty:
    st.dataframe(top_q, use_container_width=True)

st.subheader("Recent conversations")
for row in df.head(20).to_dict(orient="records"):
    st.markdown(f"**Q:** {row['question']}")
    st.markdown(f"_A:_ {row['answer'][:400]}...")
    st.caption(
        f"strategy={row['strategy']} | model={row['model']} | "
        f"latency={row['response_time']:.2f}s | tokens={row['total_tokens']} | "
        f"cost=${row['cost']:.6f}"
    )
    st.divider()
