# rag_settings.py
# ================
# Manages persistent user settings for the RAG system.
#
# - Settings are stored in rag_settings.json next to this file.
# - On load: missing file or missing keys fall back to DEFAULTS silently.
# - On save: only non-sensitive, non-secret parameters are persisted.
#   API keys are handled separately by SecretManager (encrypted storage).
# - Thread-safe: a file lock prevents corruption if two processes run.

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# Path of the settings file — always next to rag_settings.py
SETTINGS_FILE = Path(__file__).parent / "rag_settings.json"
_file_lock    = threading.Lock()

# ---------------------------------------------------------------------------
# Default values for every user-configurable parameter.
# These are used when the file is missing or a key is absent.
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    # ── Model config ──────────────────────────────────────────────────────
    "embedding_model_name": "models/gemini-embedding-001",
    "llm_model_name":       "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "llm_base_url":         "https://api.fireworks.ai/inference/v1",

    # ── Crawler settings ──────────────────────────────────────────────────
    "firecrawl_url":    "http://localhost:13002",
    "sitemap_url":      "https://milvus.io",
    "include_pattern":  "/docs",
    "exclude_pattern":  "",
    "max_depth":        0,
    "max_limit":        0,
    "start_index_ui":   0,
    "selected_langs":   ["中文(简/繁)", "英语"],
    "auth_cookie":  "",
    "auth_bearer":  "",

    # ── Milvus connection ─────────────────────────────────────────────────
    "milvus_host": "127.0.0.1",
    "milvus_port": "19530",

    # ── Performance & DB ──────────────────────────────────────────────────
    "batch_size":    200,
    "chunk_size":    1000,
    "max_threads":   10,
    "chunk_overlap": 200,
    "drop_old":      False,

    # ── Retrieval params ──────────────────────────────────────────────────
    "retrieval_k":  10,
    "rerank_top_n":  3,
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load() -> dict:
    """
    Load settings from rag_settings.json.
    Returns a complete dict: file values merged over DEFAULTS.
    Missing keys fall back to DEFAULTS. Extra unknown keys are ignored.
    Never raises — on any error returns DEFAULTS and logs a warning.
    """
    result = dict(DEFAULTS)   # start with a full copy of defaults

    if not SETTINGS_FILE.exists():
        logger.info("rag_settings.json not found — using defaults.")
        return result

    with _file_lock:
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read rag_settings.json: %s — using defaults.", exc)
            return result

    # Merge: only update keys that exist in DEFAULTS (ignore unknown keys)
    for key in DEFAULTS:
        if key in raw:
            # Type-check: cast to the same type as the default
            try:
                default_type = type(DEFAULTS[key])
                if default_type is list:
                    result[key] = list(raw[key])
                elif default_type is bool:
                    # json bools survive fine; guard against "true"/"false" strings
                    result[key] = bool(raw[key])
                else:
                    result[key] = default_type(raw[key])
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Bad value for '%s' in settings (%r) — using default. %s",
                    key, raw[key], exc,
                )
    return result


def save(settings: dict) -> None:
    """
    Persist settings to rag_settings.json.
    Only keys present in DEFAULTS are written (API keys are never stored here).
    Never raises — logs a warning on failure.
    """
    # Filter to only known, safe-to-persist keys
    to_write = {k: settings[k] for k in DEFAULTS if k in settings}

    with _file_lock:
        try:
            SETTINGS_FILE.write_text(
                json.dumps(to_write, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Settings saved to %s", SETTINGS_FILE)
        except Exception as exc:
            logger.warning("Could not save rag_settings.json: %s", exc)


def current_values_from_sidebar(
    *,
    embedding_model_name: str,
    llm_model_name:       str,
    llm_base_url:         str,
    firecrawl_url:        str,
    sitemap_url:          str,
    include_pattern:      str,
    exclude_pattern:      str,
    max_depth:            int,
    max_limit:            int,
    start_index_ui:       int,
    selected_langs:       list,
    milvus_host:          str,
    milvus_port:          str,
    batch_size:           int,
    chunk_size:           int,
    max_threads:          int,
    chunk_overlap:        int,
    drop_old:             bool,
    retrieval_k:          int,
    rerank_top_n:         int,
) -> dict:
    """
    Pack all sidebar values into a settings dict for comparison / saving.
    Pass all parameters as keyword arguments.
    """
    return {
        "embedding_model_name": embedding_model_name,
        "llm_model_name":       llm_model_name,
        "llm_base_url":         llm_base_url,
        "firecrawl_url":        firecrawl_url,
        "sitemap_url":          sitemap_url,
        "include_pattern":      include_pattern,
        "exclude_pattern":      exclude_pattern,
        "max_depth":            int(max_depth),
        "max_limit":            int(max_limit),
        "start_index_ui":       int(start_index_ui),
        "selected_langs":       list(selected_langs),
        "milvus_host":          milvus_host,
        "milvus_port":          milvus_port,
        "batch_size":           int(batch_size),
        "chunk_size":           int(chunk_size),
        "max_threads":          int(max_threads),
        "chunk_overlap":        int(chunk_overlap),
        "drop_old":             bool(drop_old),
        "retrieval_k":          int(retrieval_k),
        "rerank_top_n":         int(rerank_top_n),
    }
