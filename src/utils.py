# utils.py
import os
from dotenv import load_dotenv

load_dotenv()


def get_secret(key, default=None):
    """Read a secret from environment/.env (local) or st.secrets (Streamlit Cloud)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return default