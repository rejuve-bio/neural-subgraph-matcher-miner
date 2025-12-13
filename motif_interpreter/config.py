"""
Configuration and constants
"""
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.absolute()


try:
    from dotenv import load_dotenv
    # Load from project root
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    print("python-dotenv not installed")
    pass
except Exception as e:
    print(f"Could not load .env: {e}")

# Gemini config
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
USE_GEMINI = bool(GEMINI_KEY)

# Debug print
if USE_GEMINI:
    print(f"Gemini API key found")
else:
    print("Gemini API key NOT found in environment")
   

# Default paths
DEFAULT_PATH = "./results/patterns_all_instances.json"

# Model configurations
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_CONTEXT_LIMIT = 20