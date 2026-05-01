from __future__ import annotations

import html
import re
from typing import Any

from classifier import (
    classify_request_type,
    detect_company,
    detect_escalation,
    detect_out_of_scope,
)
from llm_client import GeminiClient
from models import EscalationSignal, RetrievalResult
from prompt_builder import OUTPUT_SCHEMA, build_prompt
from retriever import HybridRetriever


ESCALATION_TEMPLATES = {
    "fraud_suspected": (
        "We take reports of fraud and identity theft seriously. "
        "Your ticket has been escalated to our security team as a priority. "
        "Please avoid sharing additional sensitive card or account details in this channel."
    ),
    "legal_threat": (
        "Your request includes legal or compliance concerns and needs specialist review. "
        "Your ticket has been escalated to the appropriate team."
    ),
    "account_locked": (
        "Account access issues require identity verification. "
        "Your ticket has been escalated to the account recovery team for secure follow-up."
    ),
    "payment_dispute": (
        "Payment disputes need direct specialist review. "
        "Your ticket has been escalated to the billing team for follow-up."
    ),
    "exam_integrity": (
        "Assessment integrity concerns require human review. "
        "Your ticket has been escalated to the trust and safety team."
    ),
    "assessment_live": (
        "This appears time-sensitive. "
        "Your ticket has been escalated to priority support for urgent handling."
    ),
    "minor_child": (
        "This case needs careful review by a specialist. "
        "Your ticket has been escalated to the appropriate support team."
    ),
    "medical_emergency": (
        "Your message indicates an emergency and requires immediate specialist handling. "
        "Your ticket has been escalated right away."
    ),
    "prompt_injection": (
        "We cannot process this request as submitted. "
        "If you need product support, please restate your request clearly. "
        "Your ticket has been escalated for review."
    ),
    "security_disclosure": (
        "Thank you for reporting this security issue responsibly. "
        "Your ticket has been escalated to the security team for review."
    ),
    "generic": (
        "Your query requires specialist assistance. "
        "Your ticket has been escalated and a human team member will follow up."
    ),
}

PRODUCT_AREA_BY_FLAG = {
    "fraud_suspected": "Fraud & security",
    "legal_threat": "Privacy & compliance",
    "account_locked": "Account & access",
    "payment_dispute": "Transaction dispute",
    "exam_integrity": "Assessment integrity",
    "assessment_live": "Assessment support",
    "minor_child": "Account review",
    "medical_emergency": "Safety escalation",
    "prompt_injection": "Security review",
    "security_disclosure": "Bug bounty",
}

DOMAIN_DEFAULT_PRODUCT_AREA = {
    "hackerrank": "Assessment support",
    "claude": "Claude.ai usage",
    "visa": "Card support",
    "generic": "Unclear / insufficient info",
}


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _shorten_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _normalize_output(output: dict[str, Any], request_type_hint: str) -> dict[str, str]:
    status = str(output.get("status", "")).strip().lower()
    if status not in {"replied", "escalated"}:
        status = "escalated"

    request_type = str(output.get("request_type", "")).strip().lower()
    if request_type not in {"product_issue", "feature_request", "bug", "invalid"}:
        request_type = request_type_hint
    if request_type not in {"product_issue", "feature_request", "bug", "invalid"}:
        request_type = "product_issue"

    product_area = str(output.get("product_area", "")).strip()
    if not product_area:
        product_area = "Unclear / insufficient info"

    response = str(output.get("response", "")).strip()
    if not response:
        response = (
            "Your request needs additional review by a specialist. "
            "We have escalated this ticket for follow-up."
        )
        status = "escalated"

    justification = str(output.get("justification", "")).strip()
    if not justification:
        justification = "Model output fallback applied due to missing fields."

    return {
        "status": status,
        "request_type": request_type,
        "product_area": product_area,
        "response": response,
        "justification": justification,
    }


class SupportTriageAgent:
    def __init__(self, retriever: HybridRetriever, llm_client: GeminiClient) -> None:
        self.retriever = retriever
        self.llm_client = llm_client

    def _derive_product_area(
        self,
        detected_company: str,
        retrieval: RetrievalResult,
        request_type_hint: str,
    ) -> str:
        if request_type_hint == "feature_request":
            return "Feature request"
        if request_type_hint == "bug":
            return "Bug report"
        if retrieval.hits:
            filename = retrieval.hits[0].chunk.filename.lower()
            if "billing" in filename or "subscription" in filename:
                return "Billing & plans"
            if "account" in filename or "access" in filename:
                return "Account & access"
            if "privacy" in filename or "crawl" in filename:
                return "Privacy & data"
            if "assessment" in filename or "test" in filename:
                return "Assessment support"
            if "fraud" in filename or "stolen" in filename:
                return "Fraud & security"
        return DOMAIN_DEFAULT_PRODUCT_AREA.get(detected_company, "Unclear / insufficient info")

    def _fallback_grounded_reply(
        self,
        issue: str,
        detected_company: str,
        request_type_hint: str,
        retrieval: RetrievalResult,
    ) -> dict[str, str]:
        if not retrieval.hits:
            return {
                "status": "escalated",
                "product_area": "Unclear / insufficient info",
                "response": ESCALATION_TEMPLATES["generic"],
                "justification": "No relevant corpus passages found.",
                "request_type": request_type_hint,
            }

        issue_tokens = _tokenize(issue)
        picked: list[str] = []
        for hit in retrieval.hits[:3]:
            for sentence in _split_sentences(hit.chunk.text):
                sent_tokens = _tokenize(sentence)
                if issue_tokens and sent_tokens.intersection(issue_tokens):
                    picked.append(sentence)
                if len(picked) >= 3:
                    break
            if len(picked) >= 3:
                break

        if not picked:
            picked = _split_sentences(retrieval.hits[0].chunk.text)[:2]

        answer_body = " ".join(picked).strip()
        answer_body = _shorten_words(answer_body, 140)
        if not answer_body:
            return {
                "status": "escalated",
                "product_area": "Unclear / insufficient info",
                "response": ESCALATION_TEMPLATES["generic"],
                "justification": "Retrieved text could not be converted into a safe answer.",
                "request_type": request_type_hint,
            }

        response = (
            "Based on the available support guidance, "
            + answer_body
            + " If this does not resolve your case, please reply with additional details."
        )

        status = "replied" if retrieval.confidence == "high" else "escalated"
        if request_type_hint == "bug" and detected_company == "generic":
            status = "escalated"

        return {
            "status": status,
            "product_area": self._derive_product_area(
                detected_company=detected_company,
                retrieval=retrieval,
                request_type_hint=request_type_hint,
            ),
            "response": _shorten_words(response, 190),
            "justification": (
                f"Fallback grounded response using {min(3, len(retrieval.hits))} "
                f"retrieved passages with {retrieval.confidence} confidence."
            ),
            "request_type": request_type_hint,
        }

    def _hard_escalation_response(
        self,
        signals: EscalationSignal,
        request_type_hint: str,
    ) -> dict[str, str]:
        flag = signals.hard_flags[0] if signals.hard_flags else "generic"
        template = ESCALATION_TEMPLATES.get(flag, ESCALATION_TEMPLATES["generic"])
        product_area = PRODUCT_AREA_BY_FLAG.get(flag, "Specialist escalation")
        return {
            "status": "escalated",
            "product_area": product_area,
            "response": template,
            "justification": f"Hard escalation triggered by {flag}.",
            "request_type": request_type_hint,
        }

    def _invalid_response(self) -> dict[str, str]:
        return {
            "status": "replied",
            "product_area": "Out of scope",
            "response": (
                "This request is outside supported topics. "
                "If you need help with HackerRank, Claude, or Visa services, "
                "please share a product-related support question."
            ),
            "justification": "Out-of-scope or insufficiently actionable query.",
            "request_type": "invalid",
        }

    def _low_confidence_soft_flag_response(
        self,
        request_type_hint: str,
        signals: EscalationSignal,
    ) -> dict[str, str]:
        response = ESCALATION_TEMPLATES["generic"]
        return {
            "status": "escalated",
            "product_area": "Unclear / insufficient info",
            "response": response,
            "justification": (
                "Low retrieval confidence combined with soft escalation signal: "
                + ", ".join(signals.soft_flags)
            ),
            "request_type": request_type_hint,
        }

    def process_ticket(self, issue: str, subject: str, company: str) -> dict[str, str]:
        clean_issue = _clean_text(issue)
        clean_subject = _clean_text(subject)
        clean_company = _clean_text(company)

        detected_company = detect_company(clean_issue, clean_subject, clean_company)
        request_type_hint = classify_request_type(clean_issue)
        signals = detect_escalation(f"{clean_issue} {clean_subject}")
        invalid = detect_out_of_scope(clean_issue, detected_company)

        if invalid and not signals.hard_flags:
            return self._invalid_response()

        if signals.hard_flags:
            return self._hard_escalation_response(signals, request_type_hint)

        query = f"{clean_issue} {clean_subject}".strip()[:512]
        retrieval = self.retriever.retrieve(query=query, domain_hint=detected_company, k=5)

        if retrieval.confidence == "low" and signals.soft_flags:
            return self._low_confidence_soft_flag_response(request_type_hint, signals)

        if request_type_hint == "invalid":
            return self._invalid_response()

        prompts = build_prompt(
            issue=clean_issue,
            subject=clean_subject,
            company=detected_company,
            request_type_hint=request_type_hint,
            signals=signals,
            retrieval=retrieval,
        )

        model_output: dict[str, Any]
        if self.llm_client.available:
            try:
                model_output = self.llm_client.generate_json(
                    system_prompt=prompts["system"],
                    user_prompt=prompts["user"],
                    response_schema=OUTPUT_SCHEMA,
                )
            except Exception:
                return self._fallback_grounded_reply(
                    issue=clean_issue,
                    detected_company=detected_company,
                    request_type_hint=request_type_hint,
                    retrieval=retrieval,
                )
        else:
            return self._fallback_grounded_reply(
                issue=clean_issue,
                detected_company=detected_company,
                request_type_hint=request_type_hint,
                retrieval=retrieval,
            )

        normalized = _normalize_output(model_output, request_type_hint=request_type_hint)
        if detect_escalation(f"{normalized['response']} {clean_issue}").hard_flags:
            return self._hard_escalation_response(
                detect_escalation(f"{normalized['response']} {clean_issue}"),
                normalized["request_type"],
            )
        return normalized
