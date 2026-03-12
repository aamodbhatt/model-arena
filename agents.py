"""Agent definitions and prompt helpers for ModelArena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Agent:
    key: str
    name: str
    role: str
    tag: str
    emoji: str
    personality: str
    color: str
    system_prompt: str


AGENTS: Dict[str, Agent] = {
    "strategist": Agent(
        key="strategist",
        name="Hunter Alpha",
        role="Strategist",
        tag="STRAT",
        emoji="🧭",
        personality="Calm tactician",
        color="cyan",
        system_prompt=(
            "You are Hunter Alpha, the strategist in a multi-agent arena. "
            "Produce concise, practical plans. Keep output structured and clear."
        ),
    ),
    "reasoner": Agent(
        key="reasoner",
        name="NVIDIA Nemotron 3 Super",
        role="Reasoner",
        tag="REASON",
        emoji="🧠",
        personality="Methodical solver",
        color="green",
        system_prompt=(
            "You are NVIDIA Nemotron 3 Super, the primary reasoner. "
            "Solve the task step-by-step, but keep the visible output concise."
        ),
    ),
    "critic": Agent(
        key="critic",
        name="Healer Alpha",
        role="Critic",
        tag="CRITIC",
        emoji="🛡️",
        personality="Sharp challenger",
        color="red",
        system_prompt=(
            "You are Healer Alpha, a strict but constructive critic. "
            "Find flaws, missing assumptions, and potential errors."
        ),
    ),
    "analyst": Agent(
        key="analyst",
        name="Gemini Flash Lite",
        role="Analyst",
        tag="JUDGE",
        emoji="⚖️",
        personality="Fair judge",
        color="yellow",
        system_prompt=(
            "You are Gemini Flash Lite, the analyst/judge. "
            "Evaluate prior outputs and provide the best final answer."
        ),
    ),
    "vision": Agent(
        key="vision",
        name="Qwen3.5-VL",
        role="Vision Interpreter",
        tag="VISION",
        emoji="👁️",
        personality="Visual detective",
        color="magenta",
        system_prompt=(
            "You are Qwen3.5-VL, the vision interpreter. "
            "Describe visual evidence clearly and relate it to the question."
        ),
    ),
}


def get_agent(agent_key: str) -> Agent:
    return AGENTS[agent_key]


def summarize_entries(entries: Iterable[dict], max_chars_per_entry: int = 1000) -> str:
    lines: List[str] = []
    for item in entries:
        text = item["content"].strip()
        if len(text) > max_chars_per_entry:
            text = text[: max_chars_per_entry - 3] + "..."
        lines.append(f'{item["name"]} ({item["role"]}):\n{text}')
    return "\n\n".join(lines) if lines else "No prior agent messages."


def default_pipeline_prompts(question: str, history: List[dict]) -> List[tuple[str, str]]:
    context = summarize_entries(history)
    return [
        (
            "strategist",
            (
                "User question:\n"
                f"{question}\n\n"
                "Provide a concise strategy with sections: Goal, Plan, Risks."
            ),
        ),
        (
            "reasoner",
            (
                "User question:\n"
                f"{question}\n\n"
                "Prior context:\n"
                f"{context}\n\n"
                "Attempt a clear solution with sections: Approach, Solution, Checks."
            ),
        ),
        (
            "critic",
            (
                "Review the current reasoning for this question and critique it.\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Use sections: Strengths, Weaknesses, Fixes."
            ),
        ),
        (
            "reasoner",
            (
                "Revise the solution based on critique.\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Use sections: Revisions, Updated Answer, Confidence."
            ),
        ),
        (
            "analyst",
            (
                "Judge the full discussion and produce the best final answer.\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Use sections: Verdict, Final Answer, Why This Is Best."
            ),
        ),
    ]


def debate_pipeline_prompts(question: str, history: List[dict]) -> List[tuple[str, str]]:
    context = summarize_entries(history)
    return [
        (
            "reasoner",
            (
                "Provide an initial answer.\n"
                f"Question:\n{question}\n\n"
                "Use sections: Approach, Answer."
            ),
        ),
        (
            "critic",
            (
                "Critique the reasoner's answer.\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Use sections: Errors, Missing Points, Recommended Fix."
            ),
        ),
        (
            "reasoner",
            (
                "Revise your answer after critique.\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Use sections: Changes Made, Revised Answer."
            ),
        ),
        (
            "analyst",
            (
                "Judge the debate and provide the final answer.\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Use sections: Decision, Final Answer."
            ),
        ),
    ]


def committee_member_order() -> List[str]:
    return ["strategist", "reasoner", "critic", "analyst", "vision"]
