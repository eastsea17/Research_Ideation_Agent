import os

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model Names
MODEL_EMBEDDING = "nomic-embed-text:latest"

# --- MODEL_GENERATOR Configuration ---
# Select one of the following options:

# Option 1: Local Ollama Models
#MODEL_GENERATOR = "deepseek-r1:14b"
# MODEL_GENERATOR = "ministral-3:8b"

# Option 2: Ollama Cloud Models
MODEL_GENERATOR = "deepseek-v3.1:671b-cloud"
# MODEL_GENERATOR = "gpt-oss:120b-cloud"


# --- MODEL_EVALUATOR Configuration ---
# Select one of the following options:

# Option 1: Local Ollama Models
MODEL_EVALUATOR = "gpt-oss:20b"

# Option 2: Ollama Cloud Models
# MODEL_EVALUATOR = "deepseek-v3.1:671b-cloud"
# MODEL_EVALUATOR = "gpt-oss:120b-cloud"

# Vector DB Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "research_papers"
VECTOR_DB_BATCH_SIZE = 1
VECTOR_DB_SEARCH_K = 10

# Model Parameters
GENERATOR_TEMPERATURE = 0.3
EVALUATOR_TEMPERATURE = 0.3
TRANSLATOR_TEMPERATURE = 0.3

# OpenAlex API Configuration
OPENALEX_API_URL = "https://api.openalex.org/works"
USER_AGENT_EMAIL = "test@example.com" # Replace with user's email if available for polite pool

# Default Arguments
DEFAULT_PAPER_LIMIT = 200
DEFAULT_TOPIC_COUNT = 5

# Output Directory Configuration
OUTPUT_CSV_DIR = "./results/csv"
OUTPUT_REPORT_DIR = "./results"


