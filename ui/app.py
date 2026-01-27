import streamlit as st
import requests
import os

# --- Configuration ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def query_backend(question: str):
    """Handles the API communication logic."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {e}")
        return None

def display_metrics(data):
    """Renders the evaluation metrics in a structured layout."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Similarity Score", f"{data['evaluation']['similarity_score']:.2f}")
    with col2:
        status = "✅ Reliable" if data["is_reliable"] else "❌ Uncertain"
        st.metric("Status", status)

def main():
    """Main UI layout and logic."""
    st.set_page_config(page_title="RAG Pipeline Demo", page_icon="💼")
    st.title("RAG Q&A Pipeline Demo")

    # Input section
    question = st.text_input("Type your question:", placeholder="Ex.: When was Apple founded?")

    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question.")
            return

        with st.spinner("Analyzing documents..."):
            data = query_backend(question)
            
            if data:
                st.subheader("Answer:")
                if data["is_reliable"]:
                    st.success(data["answer"])
                else:
                    st.warning(f"⚠️ Low confidence answer: {data['answer']}")

                # Visualization components
                display_metrics(data)

                with st.expander("Evaluation details:"):
                    st.write(data["evaluation"]["judge_feedback"])

                with st.expander("Source Context:"):
                    st.info(data["context"])

# Standard Python entry point
if __name__ == "__main__":
    main()