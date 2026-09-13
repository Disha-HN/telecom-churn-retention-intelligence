"""
Configuration module for the Telecom Customer Churn Agent.

The module supports both local development and cloud deployment.

Local development:
    .env -> GROQ_API_KEY

Streamlit deployment:
    Streamlit Secrets -> GROQ_API_KEY
"""

import os

from dotenv import load_dotenv


# -------------------------------------------------------------------
# LOAD LOCAL ENVIRONMENT VARIABLES
# -------------------------------------------------------------------

load_dotenv()


# -------------------------------------------------------------------
# GROQ API KEY
# -------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -------------------------------------------------------------------
# STREAMLIT CLOUD FALLBACK
# -------------------------------------------------------------------

if not GROQ_API_KEY:

    try:
        import streamlit as st

        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")

    except Exception:
        GROQ_API_KEY = None


# -------------------------------------------------------------------
# VALIDATE API KEY
# -------------------------------------------------------------------

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "For local development, add it to .env. "
        "For Streamlit Cloud, add it to Streamlit Secrets."
    )


# -------------------------------------------------------------------
# MODEL CONFIGURATION
# -------------------------------------------------------------------

GROQ_MODEL = "openai/gpt-oss-20b"


# -------------------------------------------------------------------
# AGENT CONFIGURATION
# -------------------------------------------------------------------

TEMPERATURE = 0