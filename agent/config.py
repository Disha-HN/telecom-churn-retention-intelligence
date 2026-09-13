"""
Configuration module for the Telecom Customer Churn Agent.

This module:
1. Loads environment variables from the .env file.
2. Retrieves the Groq API key.
3. Stores basic project configuration.

Keeping configuration separate makes the project easier to maintain
and prevents API keys from being hard-coded inside the agent code.
"""

# -------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------

import os

from dotenv import load_dotenv


# -------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# -------------------------------------------------------------------

# Load variables from the .env file located in the project root.
load_dotenv()


# -------------------------------------------------------------------
# GROQ API CONFIGURATION
# -------------------------------------------------------------------

# Read the Groq API key from the environment.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# -------------------------------------------------------------------
# VALIDATE API KEY
# -------------------------------------------------------------------

# Stop the application early if the API key is missing.
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY was not found. "
        "Please create a .env file in the project root "
        "and add: GROQ_API_KEY=your_api_key"
    )


# -------------------------------------------------------------------
# MODEL CONFIGURATION
# -------------------------------------------------------------------

# Groq model used by the conversational agent.
GROQ_MODEL = "openai/gpt-oss-20b"


# -------------------------------------------------------------------
# AGENT CONFIGURATION
# -------------------------------------------------------------------

# Temperature controls how deterministic the LLM responses are.
# A low value is preferred for business/analytics applications.
TEMPERATURE = 0