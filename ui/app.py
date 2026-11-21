import html
import os
import re
import streamlit as st
import requests

from styles import CUSTOM_CSS

# --- API Client Layer ---


class RAGApiClient:
    """
    A generic client to handle communication with the RAG FastAPI backend.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    def ask_question(self, question: str):
        """Sends a query to the backend and returns the JSON response."""
        response = requests.post(
            f"{self.base_url}/ask",
            json={"question": question},
        )
        response.raise_for_status()
        return response.json()

    def ingest_data(self):
        """Triggers the ingestion process on the backend."""
        response = requests.post(f"{self.base_url}/ingest", timeout=300)
        response.raise_for_status()
        return response.json()


def extract_graph_nodes_from_context(context: str) -> list[str]:
    """Extract unique graph node names from the [Graph Facts] context block."""
    nodes = set()
    in_graph_block = False

    for line in context.splitlines():
        stripped = line.strip()
        if stripped == "[Graph Facts]":
            in_graph_block = True
            continue
        if in_graph_block and stripped.startswith("["):
            break

        if not in_graph_block:
            continue

        match = re.match(r"^-\s*(.+?)\s+--\[(.+?)\]-->\s*(.+?)$", stripped)
        if match:
            head = match.group(1).strip()
            tail = match.group(3).strip()
            if head:
                nodes.add(head)
            if tail:
                nodes.add(tail)

    return sorted(nodes)


def render_graph_node_chips(nodes: list[str]):
    if not nodes:
        st.caption("No graph nodes were returned for this query.")
        return

    chips = "".join(
        f'<span class="graph-chip">{html.escape(node)}</span>'
        for node in nodes
    )
    st.markdown(f'<div class="graph-chip-container">{chips}</div>', unsafe_allow_html=True)


def main():
    """Orchestrates the Streamlit UI and user interactions."""
    st.set_page_config(
        page_title="RAG Q&A",
        page_icon="◇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar: backend and ingest
    with st.sidebar:
        st.markdown("### Settings")
        backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        st.caption("Backend")
        st.code(backend_url, language=None)
        st.markdown("---")
        if st.button("Ingest documents", type="secondary", use_container_width=True):
            client = RAGApiClient(backend_url)
            with st.spinner("Ingesting…"):
                try:
                    out = client.ingest_data()
                    st.success("Ingest completed.")
                    if out:
                        st.json(out)
                except requests.exceptions.Timeout:
                    st.error("Ingest timed out.")
                except Exception as e:
                    st.error(str(e))

    client = RAGApiClient(backend_url)

    # Main content
    st.markdown("# RAG Q&A")
    st.markdown(
        '<p class="hero-tagline">Ask questions against your indexed documents. Answers are generated with Gemini and self-evaluation.</p>',
        unsafe_allow_html=True,
    )

    with st.form("query_form", clear_on_submit=False):
        question = st.text_input(
            "Question",
            placeholder="e.g. What are the main conclusions of the report?",
            label_visibility="collapsed",
        )
        submit = st.form_submit_button("Ask")

    if submit:
        if not question.strip():
            st.warning("Please enter a question.")
            return

        with st.spinner("Thinking…"):
            try:
                data = client.ask_question(question)

                # Answer card
                safe_answer = html.escape(data["answer"]).replace("\n", "<br>")
                st.markdown(
                    f'<div class="result-card"><div class="label">Answer</div><div class="value">{safe_answer}</div></div>',
                    unsafe_allow_html=True,
                )

                # Metric
                score = data["evaluation"]["similarity_score"]
                mc1, mc2 = st.columns([1, 3])
                with mc1:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{score:.2f}</div><div class="metric-label">Similarity score</div></div>',
                        unsafe_allow_html=True,
                    )

                retrieval = data.get("retrieval", {})
                graph_used = bool(retrieval.get("graph_used", False))
                graph_nodes = retrieval.get("graph_nodes", []) or extract_graph_nodes_from_context(data.get("context", ""))

                gc1, gc2, gc3, gc4 = st.columns(4)
                with gc1:
                    st.markdown(
                        f'<div class="graph-status-card"><div class="graph-state-{"active" if graph_used else "inactive"}">{"Active" if graph_used else "Inactive"}</div><div class="metric-label">Graph status</div></div>',
                        unsafe_allow_html=True,
                    )
                with gc2:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{retrieval.get("vector_hits", 0)}</div><div class="metric-label">Vector hits</div></div>',
                        unsafe_allow_html=True,
                    )
                with gc3:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{retrieval.get("graph_edges", 0)}</div><div class="metric-label">Graph edges</div></div>',
                        unsafe_allow_html=True,
                    )
                with gc4:
                    st.markdown(
                        f'<div class="metric-card"><div class="metric-value">{retrieval.get("seed_entities", 0)}</div><div class="metric-label">Seed entities</div></div>',
                        unsafe_allow_html=True,
                    )

                with st.expander("Graph nodes"):
                    render_graph_node_chips(graph_nodes)

                with st.expander("LLM-as-a-Judge feedback"):
                    st.write(data["evaluation"]["judge_feedback"])

                with st.expander("Retrieved context"):
                    st.write(data["context"])

            except requests.exceptions.Timeout:
                st.error(
                    "The request timed out. The Gemini request is taking too long to respond."
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the backend. Is the API running?")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    # Empty state hint when no query yet
    if not submit:
        st.markdown(
            '<p style="color:#64748b; font-size:0.9rem; margin-top:2rem;">Enter a question above and click **Ask** to query your documents.</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
