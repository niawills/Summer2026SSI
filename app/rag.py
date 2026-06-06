import os
import json
import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

EMBED_MODEL_DEFAULT = "text-embedding-3-small"

# ------------------------
# Utilities
# ------------------------

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v

def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    out: Dict[str, Any] = {}
    for k in ("id", "score", "metadata"):
        if hasattr(obj, k):
            out[k] = getattr(obj, k)
    return out

def _extract_json_object(text: str) -> str:
    t = (text or "").strip()

    # Strip fenced blocks
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    # Slice first {...last}
    first = t.find("{")
    last = t.rfind("}")
    if first != -1 and last != -1 and last > first:
        t = t[first:last + 1].strip()

    return t

def wants_workflow_output(query: str) -> bool:
    """
    Only generate SMS + ops when user intent is workflow/action-oriented.
    """
    q = (query or "").lower()
    triggers = [
        "draft", "sms", "text message", "respond", "reply", "apologize",
        "what should we do", "recommend", "recommendation", "action plan",
        "operations", "ops", "improve", "fix", "address", "next steps"
    ]
    return any(t in q for t in triggers)

def embed_text(client: OpenAI, text: str, embed_model: str) -> List[float]:
    return client.embeddings.create(model=embed_model, input=[text]).data[0].embedding

# ------------------------
# Retrieval + aggregation
# ------------------------

def retrieve(
    query: str,
    top_k: int = 8,
    exclude_owner_responses: bool = True,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    openai_key = _env("OPENAI_API_KEY")
    pinecone_key = _env("PINECONE_API_KEY")
    # Always use the configured main Pinecone index.
    index_name = os.getenv("PINECONE_INDEX") or index_name
    if not index_name:
        raise RuntimeError("Missing Pinecone index name. Set PINECONE_INDEX.")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", EMBED_MODEL_DEFAULT)

    client = OpenAI(api_key=openai_key)
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    qvec = embed_text(client, query, embed_model)

    filt = {"is_owner_response": False} if exclude_owner_responses else None

    res = index.query(
        vector=qvec,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        filter=filt,
        namespace=namespace,
    )

    matches = getattr(res, "matches", None) or (res.get("matches", []) if isinstance(res, dict) else []) or []
    out: List[Dict[str, Any]] = []

    for m in matches:
        mdict = _as_dict(m)
        md = mdict.get("metadata", {}) or {}

        out.append({
            "id": mdict.get("id"),
            "score": float(mdict.get("score", 0.0)),
            "text": md.get("text", ""),
            "rating": md.get("rating"),
            "date": md.get("date"),
            "themes": md.get("themes", []),
            "compound": md.get("compound"),
            "review_id": md.get("review_id"),
        })

    return out

def aggregate_contexts(
    contexts: List[Dict[str, Any]],
    min_recurring_reviews: int = 2,
) -> Dict[str, Any]:
    """
    Enforce recurrence using UNIQUE review counts (not chunk counts).
    Produces:
      - theme_stats with unique_review_count + avg_sentiment
      - recurring_issues (theme-like issues) and isolated_issues
      - sentiment calibration counts
      - sources list
    """
    theme_to_reviews: Dict[str, set] = defaultdict(set)
    theme_to_compounds: Dict[str, List[float]] = defaultdict(list)

    pos = neg = neu = 0
    for c in contexts:
        comp = c.get("compound")
        if isinstance(comp, (int, float)):
            if comp >= 0.25:
                pos += 1
            elif comp <= -0.25:
                neg += 1
            else:
                neu += 1

        rid = c.get("review_id") or c.get("id")
        for th in (c.get("themes") or []):
            theme_to_reviews[th].add(rid)
            if isinstance(comp, (int, float)):
                theme_to_compounds[th].append(float(comp))

    theme_stats = []
    for th, rset in theme_to_reviews.items():
        comps = theme_to_compounds.get(th, [])
        avg_comp = sum(comps) / len(comps) if comps else None
        theme_stats.append({
            "theme": th,
            "unique_review_count": len(rset),
            "avg_compound_sentiment": avg_comp,
        })

    # Sort: most recurring first
    theme_stats.sort(key=lambda x: x["unique_review_count"], reverse=True)

    recurring = []
    isolated = []
    for ts in theme_stats:
        item = {
            "issue": ts["theme"],
            "unique_review_count": ts["unique_review_count"],
        }
        if ts["unique_review_count"] >= min_recurring_reviews:
            recurring.append(item)
        else:
            isolated.append(item)

    sources = [c["id"] for c in contexts if c.get("id")]

    return {
        "theme_stats": theme_stats,
        "recurring_issues_by_rule": recurring,
        "isolated_issues_by_rule": isolated,
        "sentiment_counts": {"positive_chunks": pos, "neutral_chunks": neu, "negative_chunks": neg, "total_chunks": len(contexts)},
        "sources": sources,
    }

# ------------------------
# Prompting
# ------------------------

def build_prompt(
    query: str,
    contexts: List[Dict[str, Any]],
    agg: Dict[str, Any],
    workflow: bool,
    max_chunk_chars: int = 650
) -> List[Dict[str, str]]:
    ctx_lines = []
    for c in contexts:
        txt = (c.get("text") or "").replace("\n", " ").strip()
        if len(txt) > max_chunk_chars:
            txt = txt[:max_chunk_chars].rstrip() + "…"

        ctx_lines.append(
            f"- [chunk_id={c['id']}] (review_id={c.get('review_id')}, rating={c.get('rating')}, date={c.get('date')}, score={c.get('score'):.3f}, compound={c.get('compound')}) {txt}"
        )

    ctx_block = "\n".join(ctx_lines)

    sent = agg.get("sentiment_counts", {})
    recurring_rule = agg.get("recurring_issues_by_rule", [])
    isolated_rule = agg.get("isolated_issues_by_rule", [])
    theme_stats = agg.get("theme_stats", [])

    system = (
        "You are a customer feedback analyst.\n"
        "You MUST respond with a single JSON object (no markdown).\n"
        "Use ONLY the provided review chunks as evidence.\n"
        "If the retrieved chunks do not support a claim, say: "
        "\"Not enough evidence in retrieved reviews.\""
    )

    # Same JSON schema always (stable for Streamlit),
    # but when workflow=False: sms_draft + ops_recommendations should be empty/minimal.
    user = f"""
USER REQUEST:
{query}

SENTIMENT CALIBRATION (from retrieved chunks):
- positive_chunks: {sent.get("positive_chunks")}
- neutral_chunks: {sent.get("neutral_chunks")}
- negative_chunks: {sent.get("negative_chunks")}
- total_chunks: {sent.get("total_chunks")}

THEME RECURRENCE (computed by code using UNIQUE review counts):
theme_stats: {json.dumps(theme_stats, ensure_ascii=False)}
recurring_by_rule(min_unique_reviews): {json.dumps(recurring_rule, ensure_ascii=False)}
isolated_by_rule: {json.dumps(isolated_rule, ensure_ascii=False)}

RETRIEVED REVIEW CHUNKS:
{ctx_block}

Return ONLY valid JSON with this schema:

{{
  "answer_summary": "string",
  "top_themes": [
    {{
      "theme": "string",
      "sentiment": "positive|mixed|negative",
      "evidence": [{{"chunk_id": "string", "excerpt": "string"}}]
    }}
  ],
  "overall_sentiment": {{
    "label": "positive|mixed|negative",
    "rationale": "string"
  }},
  "recurring_issues": [
    {{
      "issue": "string",
      "unique_review_count": 0,
      "evidence_chunk_ids": ["string"]
    }}
  ],
  "isolated_issues": [
    {{
      "issue": "string",
      "unique_review_count": 0,
      "evidence_chunk_ids": ["string"]
    }}
  ],
  "sms_draft": {{
    "messages": ["string"],
    "character_counts": [0]
  }},
  "ops_recommendations": [
    {{"recommendation": "string", "grounding_chunk_ids": ["string"]}}
  ],
  "sources": ["string"]
}}

Rules:
- Provide 3–5 themes max.
- Evidence excerpts must be <= 25 words and MUST come from chunk text.
- Calibrate claims: never say "all reviews" unless supported by retrieved chunks + sentiment counts.
- recurring_issues MUST align with recurring_by_rule (unique_review_count >= threshold).
- isolated_issues MUST align with isolated_by_rule (unique_review_count < threshold).
- If workflow is FALSE, set sms_draft.messages = [] and ops_recommendations = [].
- If workflow is TRUE, sms_draft: 2 messages max, warm/professional, no emojis, each <=160 chars.
- ops_recommendations: 2–3 items max, each grounded to chunk_ids.
- Always include sources list (chunk IDs used).
Workflow flag for this request: {str(workflow)}
""".strip()

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ------------------------
# Main entry
# ------------------------

def generate_grounded_response(
    query: str,
    top_k: int = 8,
    min_recurring_reviews: int = 2,
    include_debug: bool = False,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    openai_key = _env("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=openai_key)

    workflow = wants_workflow_output(query)

    contexts = retrieve(
        query,
        top_k=top_k,
        exclude_owner_responses=True,
        index_name=index_name,
        namespace=namespace,
    )
    agg = aggregate_contexts(contexts, min_recurring_reviews=min_recurring_reviews)
    messages = build_prompt(query, contexts, agg, workflow=workflow)

    resp = client.responses.create(
        model=model,
        input=messages,
        text={"format": {"type": "json_object"}},
        temperature=0,
    )

    raw = (resp.output_text or "").strip()
    cooked = _extract_json_object(raw)

    try:
        out = json.loads(cooked)
    except json.JSONDecodeError as e:
        return {
            "error": "Model did not return valid JSON",
            "raw_output": raw[:4000],
            "exception": str(e),
        }

    # Force rule alignment (safety net)
    out.setdefault("sources", agg.get("sources", []))

    if not workflow:
        out["sms_draft"] = {"messages": [], "character_counts": []}
        out["ops_recommendations"] = []

    if include_debug:
        out["debug"] = {
            "workflow": workflow,
            "sentiment_counts": agg.get("sentiment_counts"),
            "theme_stats": agg.get("theme_stats")[:10],
            "recurring_by_rule": agg.get("recurring_issues_by_rule"),
            "isolated_by_rule": agg.get("isolated_issues_by_rule"),
            "top_k": top_k,
            "min_recurring_reviews": min_recurring_reviews,
        }

    return out
