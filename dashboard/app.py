"""Cost-vs-Correctness dashboard.

    pip install "agent-memory-lab[dashboard]"
    streamlit run dashboard/app.py

Shows the thing end-to-end F1 hides: a backend can post a fine update-EM while
its staleness rate quietly corrupts recall, and the cost of staying current
(edge closes, scan size) differs sharply across designs.
"""

from __future__ import annotations

import streamlit as st

from agent_memory_lab.backends import ALL_BACKENDS, BACKENDS
from agent_memory_lab.stalebench.runner import run_benchmark

st.set_page_config(page_title="Agent Memory Lab — StaleBench", layout="wide")
st.title("Agent Memory Lab — StaleBench")
st.caption("Invalidation is the unsolved primitive. Look below the final score.")

with st.sidebar:
    st.header("Run")
    backends = st.multiselect("Backends", ALL_BACKENDS, default=BACKENDS)
    episodes = st.slider("Episodes", 5, 100, 30, step=5)
    scenario = st.selectbox("Scenario", ["all", "retract", "basic"])
    provider = st.selectbox("Provider", ["mock", "mlx"])
    go = st.button("Run benchmark", type="primary")

if go and backends:
    with st.spinner("Running StaleBench..."):
        rows = run_benchmark(backends, provider=provider,
                             n_episodes=episodes, scenario=scenario)

    try:
        import pandas as pd
        df = pd.DataFrame(rows).set_index("backend")
    except ModuleNotFoundError:
        df = None

    st.subheader("Correctness — lower staleness is better")
    if df is not None:
        st.bar_chart(df[["staleness_rate", "update_em", "history_recall"]])
        st.subheader("Operational cost")
        st.bar_chart(df[["avg_store_size", "avg_scanned", "closes"]])
        st.subheader("Cost vs staleness")
        st.scatter_chart(df.reset_index(), x="avg_scanned", y="staleness_rate",
                         color="backend")
        st.subheader("Raw results")
        st.dataframe(df)
    else:
        st.table(rows)

    best = min(rows, key=lambda r: r["staleness_rate"])
    st.success(f"Lowest staleness: **{best['backend']}** "
               f"({best['staleness_rate']*100:.0f}%), "
               f"update-EM {best['update_em']*100:.0f}%, "
               f"history recall {best['history_recall']*100:.0f}%.")
else:
    st.info("Pick backends in the sidebar and hit **Run benchmark**.")
