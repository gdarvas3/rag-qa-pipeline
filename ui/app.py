import streamlit as st
import requests
import os

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

def main():
    """Orchestrates the Streamlit UI and user interactions."""
    st.set_page_config(page_title="RAG QA Demo", page_icon="💼", layout="centered")
    st.title("RAG Q&A Demo")

    # Initialize the API client
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    client = RAGApiClient(backend_url)

    # User Input
    with st.form("query_form", clear_on_submit=False):
        question = st.text_input("Question", placeholder="Ask something based on your documents.")
        submit_button = st.form_submit_button("Query")

    if submit_button:
        if not question.strip():
            st.warning("Please enter a valid question.")
            return

        with st.spinner("Processing..."):
            try:
                # Execute logic
                data = client.ask_question(question)
                
                # Display core answer
                st.subheader("System Response")
                st.success(data["answer"])

                # Render insights
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Similarity Score", f"{data['evaluation']['similarity_score']:.2f}")

                # Expandable details
                with st.expander("LLM-as-a-Judge Feedback"):
                    st.write(data["evaluation"]["judge_feedback"])

                with st.expander("Retrieved Context"):
                    st.info(data["context"])

            except requests.exceptions.Timeout:
                st.error("The request timed out. The local LLM is taking too long to generate an answer.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()