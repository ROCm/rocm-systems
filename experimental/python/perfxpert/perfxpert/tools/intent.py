"""intent — rule-based user-query classifier for air-gap routing.

Design-review C2: Root's routing must work without LLM.
This is a keyword/regex classifier that covers common user intents.
LLM (when available) can override; but this always returns SOMETHING.

Tool class: READ_ONLY.
"""

import re
from typing import Any, Dict, Literal

from perfxpert.tools._class import ToolClass, tool_class


Intent = Literal["analyze", "optimize", "verify", "explain", "help"]


_PATTERNS = [
    # verify checks must precede help so "did my patch help" routes to verify
    # (verify pattern `\bdid\b.+\b(help|...)\b` vs help's bare `\bhelp\b`)
    ("verify", [
        r"\bdid\b.+\b(help|improve|work)\b",
        r"\bcompare\b.+\brun\b",
        r"\bis this better\b",
        r"\bregression\b",
        r"\bverify\b",
    ]),
    ("help", [r"\bhelp\b", r"\bwhat can\b", r"\bhow do I use\b"]),
    ("optimize", [
        r"\b(how|what).*\b(fix|improve|optimize|speed up)\b",
        r"\bsuggest\b.*\boptimizations?\b",
        r"\bsuggest optimization\b",
        r"\brecommend\b",
        r"\boptimiz",
    ]),
    ("explain", [
        r"\bwhat (is|does|means?)\b",
        r"\bexplain\b",
        r"\bdefine\b",
    ]),
    ("analyze", [
        r"\banalyze\b",
        r"\b(why|what).*\b(slow|bottleneck|performance|fast)\b",
        r"\btrace\b",
        r"\bprofile\b",
        r"\binvestigate\b",
    ]),
]


@tool_class(ToolClass.READ_ONLY)
def classify(user_query: str) -> Dict[str, Any]:
    """Classify a user query into one of five intents via keyword matching.

    Args:
        user_query: raw user string.

    Returns:
        {"intent": Intent, "confidence": float, "matched_pattern": str,
         "warning": str (present when confidence < 0.5)}
    """
    query = user_query.lower()
    for intent_name, patterns in _PATTERNS:
        for p in patterns:
            if re.search(p, query):
                return {
                    "intent": intent_name,
                    "confidence": 0.9,
                    "matched_pattern": p,
                }

    # Default to "analyze" for ambiguous queries
    return {
        "intent": "analyze",
        "confidence": 0.3,
        "matched_pattern": None,
        "warning": "Query is ambiguous; defaulting to 'analyze'",
    }
