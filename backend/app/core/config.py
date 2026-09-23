
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "autism_child" / "Autism-Child-Data.arff"
RAG_DIR = PROJECT_ROOT / "data" / "processed" / "rag"
RAG_INDEX_PATH = RAG_DIR / "knowledge_base.faiss"
RAG_METADATA_PATH = RAG_DIR / "metadata.json"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACT_DIR = MODELS_DIR / "artifacts"
MODEL_METADATA_DIR = MODELS_DIR / "metadata"
REGISTRY_DIR = MODELS_DIR / "registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"

MODEL_NAME = "neuro_screen_logistic_regression"
MODEL_VERSION = "v2.0.0"
MODEL_STATUS = "DEMO_ACTIVE"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

SCORE_COLUMNS = [f"A{i}_Score" for i in range(1, 11)]
NUMERIC_FEATURES = [*SCORE_COLUMNS, "age"]
CATEGORICAL_FEATURES = [
    "gender", "ethnicity", "jundice", "austim",
    "contry_of_res", "used_app_before", "relation",
]
FEATURE_NAMES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
TARGET_COLUMN = "Class/ASD"
EXCLUDED_FEATURES = ["result", "age_desc"]

# The child model above covers ages 4-11 only (data/raw/autism_child).
# This second, separately trained model covers toddlers 12-36 months
# (1-3 years) using the Q-CHAT-10-Toddler instrument -- a different
# questionnaire for a different developmental stage, not a superset/
# subset of the child one, so it is kept as its own model rather than
# merged into one feature space. See docs/architecture.md.
TODDLER_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "autism_toddler" / "Toddler-Autism-Data.csv"

TODDLER_MODEL_NAME = "neuro_screen_logistic_regression_toddler"
TODDLER_MODEL_VERSION = "v1.0.0"
TODDLER_MODEL_STATUS = "DEMO_ACTIVE"

TODDLER_SCORE_COLUMNS = [f"A{i}" for i in range(1, 11)]
TODDLER_NUMERIC_FEATURES = [*TODDLER_SCORE_COLUMNS, "Age_Mons"]
TODDLER_CATEGORICAL_FEATURES = [
    "Sex", "Ethnicity", "Jaundice", "Family_mem_with_ASD", "Who completed the test",
]
TODDLER_FEATURE_NAMES = [*TODDLER_NUMERIC_FEATURES, *TODDLER_CATEGORICAL_FEATURES]
TODDLER_TARGET_COLUMN = "Class/ASD Traits "
TODDLER_EXCLUDED_FEATURES = ["Case_No", "Qchat-10-Score"]

TODDLER_AGE_MIN_YEARS = 1.0
TODDLER_AGE_MAX_YEARS = 3.0

for directory in (ARTIFACT_DIR, MODEL_METADATA_DIR, REGISTRY_DIR, RAG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Environment-driven runtime settings.

    Values fall back to safe local-dev defaults so the app still runs
    with no .env present; override via env vars prefixed NEUROSCREEN_
    (see .env.example).
    """

    model_config = SettingsConfigDict(
        env_prefix="NEUROSCREEN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'neuroscreen.db').as_posix()}"
    ollama_model: str = "gemma2:2b"
    ollama_host: str = "http://localhost:11434"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
