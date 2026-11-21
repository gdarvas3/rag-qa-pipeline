"""Custom CSS for the RAG Q&A Streamlit UI."""

CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Outfit:wght@300;400;500;600;700&display=swap');

  /* Root and main container */
  .stApp {
    background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
  }
  .main .block-container {
    max-width: 720px;
    padding-top: 2rem;
    padding-bottom: 4rem;
  }

  /* Headings */
  h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
  }
  h1 {
    font-size: 1.85rem !important;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* Hero / tagline */
  .hero-tagline {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    color: #94a3b8;
    margin: -0.5rem 0 1.5rem 0;
    letter-spacing: 0.01em;
  }

  /* Query input area */
  .stTextInput input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background-color: #1e293b !important;
  }
  .stTextInput input:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 2px rgba(13, 148, 136, 0.2) !important;
  }

  /* Cards */
  .result-card {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
  }
  .result-card .label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #0d9488;
    margin-bottom: 0.5rem;
  }
  .result-card .value {
    font-size: 1rem;
    line-height: 1.6;
    color: #e2e8f0;
  }
  .metric-card {
    font-family: 'Outfit', sans-serif;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .metric-card .metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0d9488;
  }
  .metric-card .metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.25rem;
  }

  .graph-status-card {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .graph-state-active {
    font-size: 1.15rem;
    font-weight: 600;
    color: #14b8a6;
  }
  .graph-state-inactive {
    font-size: 1.15rem;
    font-weight: 600;
    color: #64748b;
  }
  .graph-chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.75rem;
  }
  .graph-chip {
    display: inline-flex;
    align-items: center;
    background: rgba(13, 148, 136, 0.15);
    border: 1px solid rgba(13, 148, 136, 0.35);
    color: #e2e8f0;
    border-radius: 999px;
    padding: 0.4rem 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
  }

  /* Buttons */
  .stForm button[kind="primary"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
    border: none !important;
  }
  .stForm button[kind="primary"]:hover {
    background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%) !important;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid #334155;
  }
  [data-testid="stSidebar"] .stMarkdown {
    font-family: 'Outfit', sans-serif;
  }

  /* Expanders */
  .streamlit-expanderHeader {
    font-family: 'Outfit', sans-serif !important;
    background: #1e293b !important;
    border-radius: 10px !important;
  }

  /* Hide Streamlit branding (optional) */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
</style>
"""
