"""
Validator — hardcoded security guardrail, zero LLM calls.

Checks every user message for three threat categories before Marco runs:
  1. Prompt injection  — attempts to override system instructions
  2. Out-of-scope      — requests completely unrelated to travel
  3. Unsupported city  — destination not in the database

All checks are regex / keyword based and compile once at class load time.
"""

import re
from dataclasses import dataclass
from typing import Optional


# ── Result dataclass (same interface as the old LLM validator) ────────────────

@dataclass
class ValidationResult:
    approved: bool
    verdict: str          # APPROVED | BLOCKED_INJECTION | BLOCKED_SCOPE | BLOCKED_CITY
    reason: str
    rejection_message: str


# ── Rejection messages shown to the user ─────────────────────────────────────

_REJECTION_MESSAGES = {
    "BLOCKED_INJECTION": (
        "I noticed your message contains instructions trying to change my behaviour. "
        "I'm Marco, your travel planning assistant, and I can only help with "
        "flights, hotels, activities, and trip planning."
    ),
    "BLOCKED_SCOPE": (
        "That's outside what I can help with. I'm Marco, a travel planning assistant. "
        "Ask me about flights, hotels, activities, or trip costs for "
        "Paris, London, Tokyo, New York, or Berlin."
    ),
    "BLOCKED_CITY": (
        "Sorry, that destination isn't in my database yet. "
        "I currently support: Paris, London, Tokyo, New York, and Berlin. "
        "Would you like to plan a trip to one of those instead?"
    ),
}


# ── Main validator class ──────────────────────────────────────────────────────

class InputValidator:
    """
    Stateless hardcoded validator.  All patterns compile once on first use.
    Call InputValidator.validate(message) → ValidationResult.
    """

    # Cities in the database
    SUPPORTED_CITIES = frozenset({"paris", "london", "tokyo", "new york", "berlin"})

    # Common world cities NOT in the database — detected to give a helpful block
    KNOWN_UNSUPPORTED_CITIES = frozenset({
        "rome", "madrid", "amsterdam", "dubai", "bangkok", "sydney",
        "barcelona", "singapore", "istanbul", "prague", "vienna",
        "los angeles", "chicago", "toronto", "seoul", "beijing", "shanghai",
        "hong kong", "mumbai", "delhi", "cairo", "mexico city",
        "buenos aires", "johannesburg", "moscow", "athens", "lisbon",
        "florence", "venice", "milan", "brussels", "geneva", "zurich",
        "stockholm", "oslo", "copenhagen", "helsinki", "warsaw", "budapest",
        "kyoto", "osaka", "bali", "phuket", "cancun", "havana",
        "nairobi", "casablanca", "abu dhabi", "doha", "riyadh",
        "tel aviv", "jerusalem", "beirut", "karachi", "lahore",
        "lagos", "accra", "tunis", "algiers", "cape town",
    })

    # Regex patterns that signal prompt injection
    _INJECTION_PATTERNS_RAW = [
        r"ignore\s+(previous|all|your|the)\s+instructions?",
        r"disregard\s+(all|previous|your|the)\s+instructions?",
        r"new\s+system\s+prompt",
        r"your\s+new\s+role",
        r"\bact\s+as\s+(a|an|if)\b",
        r"pretend\s+(you\s+are|to\s+be)",
        r"you\s+are\s+now\s+a",
        r"\bjailbreak\b",
        r"\bDAN\b",
        r"do\s+anything\s+now",
        r"what\s+are\s+your\s+(instructions?|rules?|prompt)",
        r"(repeat|show|reveal|print|output)\s+(your\s+)?(system\s+)?(prompt|instructions?|rules?)",
        r"forget\s+(everything|all|your\s+instructions?)",
        r"override\s+(your|the)\s+(instructions?|prompt|rules?)",
        r"(developer|god|admin|sudo|root)\s+mode",
        r"bypass\s+(your|the)\s+(restrictions?|rules?|guidelines?|filters?)",
        r"from\s+now\s+on\s+(you|act|be|ignore|forget)",
        r"you\s+are\s+(actually|really|secretly|truly)\s+a",
        r"(switch|change|enter)\s+(to\s+)?(a\s+)?(different|new|unrestricted)\s+mode",
        r"simulate\s+(a|an)\s+.*(ai|assistant|bot|system)",
    ]

    # (pattern, topic_label) pairs for clearly off-topic content
    _OFF_TOPIC_PATTERNS_RAW = [
        (r"\b(solve|calculate|compute|evaluate)\s+(this\s+)?(equation|math|formula|integral|derivative|sum|problem)", "mathematics"),
        (r"\b\d+\s*[\+\-\*\/\^]\s*\d+\b", "arithmetic"),
        (r"\bwrite\s+(me\s+)?(a\s+)?(poem|essay|story|song|lyrics|novel|script|haiku|sonnet)", "creative writing"),
        (r"\b(write|generate|create|give me)\s+(some\s+)?(code|function|class|algorithm|script|program)\b", "coding"),
        (r"\b(debug|fix|review)\s+(this|my|the)\s+(code|function|script|program|bug)", "coding"),
        (r"\bwhat\s+is\s+the\s+(capital|population|president|prime\s+minister|gdp|area)\s+of\b", "general knowledge"),
        (r"\bwho\s+(is|was|invented|discovered|wrote|created|founded)\b", "general knowledge"),
        (r"\btranslate\s+(this|the|from|to|into)\b", "translation"),
        (r"\bplay\s+(a\s+)?(game|chess|quiz|trivia|riddle)\b", "games"),
        (r"\b(stock|crypto|bitcoin|ethereum|forex)\s+(market|price|trading|chart)\b", "finance"),
        (r"\b(recipe|how\s+to\s+cook|how\s+to\s+bake|ingredient|dish)\b", "cooking"),
        (r"\bsport(s)?\s+(score|result|match|standings|league)\b", "sports"),
        (r"\b(diagnosis|symptom|medicine|prescription|disease|treatment)\b", "medical"),
        (r"\b(law|legal\s+advice|is\s+it\s+legal|lawsuit|attorney)\b", "legal"),
        (r"\bthe\s+(meaning\s+of\s+life|universe|everything|big\s+bang)\b", "philosophy/science"),
    ]

    # Travel-related keywords — any match overrides the off-topic check
    _TRAVEL_KEYWORDS = frozenset({
        "flight", "flights", "hotel", "hotels", "trip", "travel", "travelling",
        "destination", "visa", "activities", "activity", "itinerary", "airport",
        "airline", "vacation", "holiday", "tourism", "tourist", "accommodation",
        "booking", "book", "ticket", "passport", "tour", "sightseeing",
        "paris", "london", "tokyo", "new york", "berlin",
        "cheapest", "budget", "cost", "price", "nights", "days",
        "plan", "help", "hi", "hello",
    })

    # Compiled pattern caches
    _injection_compiled: Optional[list] = None
    _off_topic_compiled: Optional[list] = None

    @classmethod
    def _injection_patterns(cls) -> list:
        if cls._injection_compiled is None:
            cls._injection_compiled = [
                re.compile(p, re.IGNORECASE) for p in cls._INJECTION_PATTERNS_RAW
            ]
        return cls._injection_compiled

    @classmethod
    def _off_topic_patterns(cls) -> list:
        if cls._off_topic_compiled is None:
            cls._off_topic_compiled = [
                (re.compile(p, re.IGNORECASE), label)
                for p, label in cls._OFF_TOPIC_PATTERNS_RAW
            ]
        return cls._off_topic_compiled

    @classmethod
    def validate(cls, message: str) -> ValidationResult:
        msg_lower = message.lower().strip()

        # ── 1. Injection check (highest priority) ────────────────────────────
        for pattern in cls._injection_patterns():
            if pattern.search(message):
                return ValidationResult(
                    approved=False,
                    verdict="BLOCKED_INJECTION",
                    reason="Prompt injection attempt detected.",
                    rejection_message=_REJECTION_MESSAGES["BLOCKED_INJECTION"],
                )

        # ── 2. City check ─────────────────────────────────────────────────────
        detected = cls._detect_city(msg_lower)
        if detected and detected not in cls.SUPPORTED_CITIES:
            return ValidationResult(
                approved=False,
                verdict="BLOCKED_CITY",
                reason=f"Unsupported destination: {detected}.",
                rejection_message=(
                    f"Sorry, {detected.title()} isn't in my database yet. "
                    "I currently support: Paris, London, Tokyo, New York, and Berlin. "
                    "Would you like to plan a trip to one of those instead?"
                ),
            )

        # ── 3. Off-topic check (only if no travel keyword present) ────────────
        has_travel_keyword = any(kw in msg_lower for kw in cls._TRAVEL_KEYWORDS)
        if not has_travel_keyword:
            for pattern, topic in cls._off_topic_patterns():
                if pattern.search(message):
                    return ValidationResult(
                        approved=False,
                        verdict="BLOCKED_SCOPE",
                        reason=f"Off-topic request: {topic}.",
                        rejection_message=_REJECTION_MESSAGES["BLOCKED_SCOPE"],
                    )

        # ── 4. Approved ───────────────────────────────────────────────────────
        return ValidationResult(
            approved=True,
            verdict="APPROVED",
            reason="Valid travel-related request.",
            rejection_message="",
        )

    @classmethod
    def _detect_city(cls, msg_lower: str) -> Optional[str]:
        for city in cls.SUPPORTED_CITIES:
            if city in msg_lower:
                return city
        for city in cls.KNOWN_UNSUPPORTED_CITIES:
            if city in msg_lower:
                return city
        return None


# ── Public function (called by nodes.py) ─────────────────────────────────────

def validate_input(user_message: str) -> ValidationResult:
    return InputValidator.validate(user_message)
