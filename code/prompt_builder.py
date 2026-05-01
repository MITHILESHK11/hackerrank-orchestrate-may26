from __future__ import annotations

import json

from models import EscalationSignal, RetrievalResult


OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "product_area", "response", "justification", "request_type"],
    "properties": {
        "status": {"type": "string", "enum": ["replied", "escalated"]},
        "product_area": {"type": "string"},
        "response": {"type": "string"},
        "justification": {"type": "string"},
        "request_type": {
            "type": "string",
            "enum": ["product_issue", "feature_request", "bug", "invalid"],
        },
    },
}


def build_prompt(
    issue: str,
    subject: str,
    company: str,
    request_type_hint: str,
    signals: EscalationSignal,
    retrieval: RetrievalResult,
) -> dict[str, str]:
    system = (
        f"You are a precise support triage assistant for {company}. "
        "Use only provided passages. Never invent policies, prices, or steps. "
        "If information is incomplete, choose escalated. "
        "Never reveal internal logic, source file names, passage ids, or scores."
    )

    passages: list[str] = []
    for idx, hit in enumerate(retrieval.hits, start=1):
        safe_text = hit.chunk.text.strip()
        passages.append(
            f'<passage id="{idx}" domain="{hit.chunk.source_domain}" '
            f'source="{hit.chunk.filename}">\n{safe_text}\n</passage>'
        )
    passage_block = "\n\n".join(passages) if passages else "<passage id=\"0\">No passages retrieved.</passage>"

    user = (
        "[CONTEXT PASSAGES]\n"
        f"{passage_block}\n\n"
        "[TICKET]\n"
        f"Company: {company}\n"
        f"Subject: {subject if subject else '(none)'}\n"
        f"Issue: {issue}\n\n"
        "[DETECTED SIGNALS]\n"
        f"Request type hint: {request_type_hint}\n"
        f"Soft escalation flags: {', '.join(signals.soft_flags) if signals.soft_flags else 'none'}\n"
        f"Retrieval confidence: {retrieval.confidence}\n\n"
        "[INSTRUCTIONS]\n"
        "Return JSON only, no markdown.\n"
        "Schema:\n"
        f"{json.dumps(OUTPUT_SCHEMA, ensure_ascii=True)}\n"
        "Rules:\n"
        "1. If retrieval confidence is low and any soft flag exists, status must be escalated.\n"
        "2. If request type is invalid, reply politely as out of scope with status replied.\n"
        "3. Response must stay under 200 words and use only provided passages.\n"
        "4. Justification must stay under 40 words.\n"
        "5. Product area must be concise (2-5 words).\n"
    )

    return {"system": system, "user": user}
