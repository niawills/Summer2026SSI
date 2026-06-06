import os
import json
from typing import Dict, Optional, Tuple


def load_restaurant_map() -> Dict[str, str]:
    """Load mapping of restaurant display name -> Pinecone index name.

    Priority:
      1. RESTAURANT_INDICES env var (JSON)
      2. restaurants.json file in repo root
      3. fall back to PINECONE_INDEX as single Default entry
    """
    raw = os.getenv("RESTAURANT_INDICES", "")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass

    # Try file
    try:
        root = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(root, "restaurants.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass

    # Fallback to single default
    default_idx = os.getenv("PINECONE_INDEX", "") or ""
    return {"Default": default_idx}


def validate_index_name(index_name: Optional[str]) -> Tuple[bool, str]:
    """Return (is_valid, message). Currently only checks non-empty string.
    Could be extended to probe Pinecone or enforce naming rules.
    """
    if not index_name or not isinstance(index_name, str):
        return False, "No index name provided."
    s = index_name.strip()
    if not s:
        return False, "Index name is empty after trimming."
    # Basic sanity: length and characters
    if len(s) > 100:
        return False, "Index name is unusually long."
    return True, "Looks good."
