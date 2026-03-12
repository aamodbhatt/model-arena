"""CLI entrypoint for ModelArena."""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from agents import get_agent, summarize_entries
from openrouter_client import OpenRouterClient
from ui import (
    ShowcaseSession,
    print_agent_legend,
    print_agent_panel,
    print_flow_plan,
    print_flow_status,
    print_header,
    print_info,
    print_judge_scorecard,
    print_mission,
    print_phase_banner,
    print_research_brief,
    print_step_loading,
    supports_live_layout,
    print_warning,
    prompt_question,
    show_thinking,
)


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
LOGS_DIR = BASE_DIR / "logs"
REQUIRED_MODEL_KEYS = ["strategist", "reasoner", "critic", "analyst", "vision"]
DEFAULT_REASONER_FALLBACK_CANDIDATES = [
    "deepseek/deepseek-chat-v3-0324",
    "google/gemini-2.0-flash-001",
]
DEFAULT_FREE_MODELS = {
    "strategist": "openrouter/hunter-alpha",
    "reasoner": "nvidia/nemotron-3-super-120b-a12b:free",
    "critic": "openrouter/healer-alpha",
    "analyst": "openrouter/hunter-alpha",
    "vision": "nvidia/nemotron-nano-12b-v2-vl:free",
}
DEFAULT_FREE_REASONER_FALLBACK_CANDIDATES = [
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openrouter/hunter-alpha",
    "openrouter/healer-alpha",
    "meta-llama/llama-3.3-70b-instruct:free",
]
ALWAYS_FREE_MODEL_IDS = {
    "openrouter/hunter-alpha",
    "openrouter/healer-alpha",
    "openrouter/free",
}
OUTPUT_STYLE_GUARD = (
    "Output style rules:\n"
    "- Keep it concise but complete.\n"
    "- Do not repeat the question.\n"
    "- Use plain text section labels (for example: Plan:, Answer:, Risks:).\n"
    "- Avoid markdown formatting symbols like #, ##, *, or ** in the output.\n"
    "- Avoid fenced code blocks unless explicitly required.\n"
)
ROUND_NOTE = (
    "Debate protocol:\n"
    "- Keep it compact and structured.\n"
    "- Stay within the requested template.\n"
    "- Max 120 words unless final answer needs more precision.\n"
)

TEMPERATURE_BY_AGENT = {
    "strategist": 0.3,
    "reasoner": 0.35,
    "critic": 0.25,
    "analyst": 0.2,
    "vision": 0.2,
}

MAX_TOKENS_BY_AGENT = {
    "strategist": 420,
    "reasoner": 650,
    "critic": 450,
    "analyst": 520,
    "vision": 520,
}


ANALYST_VERDICT_TEMPLATE = (
    "Round: Verdict.\n"
    "Return EXACTLY these fields with one line each:\n"
    "Correctness: <0-10>\n"
    "Clarity: <0-10>\n"
    "Evidence Quality: <0-10>\n"
    "Risk Flags: <comma-separated top risks>\n"
    "Winner: <best contributor or best round>\n"
    "Takeaway: <one sentence>\n"
    "Final Answer: <concise final answer>\n"
)


def _judge_summary_from_text(text: str) -> Dict[str, str]:
    def extract_line(label: str) -> str:
        match = re.search(rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$", text)
        return match.group(1).strip() if match else ""

    summary = {
        "correctness": extract_line("Correctness") or "-",
        "clarity": extract_line("Clarity") or "-",
        "evidence_quality": extract_line("Evidence Quality") or "-",
        "risk_flags": extract_line("Risk Flags") or "None listed",
        "winner": extract_line("Winner") or "Not specified",
        "takeaway": extract_line("Takeaway") or "No takeaway provided.",
        "final_answer": extract_line("Final Answer"),
    }
    return summary


@dataclass
class ArenaRuntime:
    client: OpenRouterClient
    models: Dict[str, str]
    mode: str
    question: str
    history: List[dict] = field(default_factory=list)
    flow: List[dict] = field(default_factory=list)
    flow_status: List[str] = field(default_factory=list)
    free_mode: bool = False
    session_notes: List[str] = field(default_factory=list)
    research_brief: str = ""

    def start_flow(self, flow: List[dict], render: bool = True) -> None:
        self.flow = flow
        self.flow_status = ["pending"] * len(flow)
        if render:
            print_flow_plan(flow)

    def finish_flow(self, render: bool = True) -> None:
        if self.flow and self.flow_status and render:
            print_flow_status(self.flow, self.flow_status, len(self.flow_status) - 1)

    def invoke(
        self,
        agent_key: str,
        prompt: str,
        image_path: Optional[str] = None,
        phase: str = "",
        step_index: int = 0,
        total_steps: int = 1,
        final_highlight: bool = False,
        showcase: Optional[ShowcaseSession] = None,
    ) -> str:
        model = self.models[agent_key]
        agent = resolve_agent_view(agent_key, model)
        styled_prompt = f"{prompt.strip()}\n\n{ROUND_NOTE}\n{OUTPUT_STYLE_GUARD}"
        usage: Dict[str, int] = {}
        started = time.perf_counter()

        if self.flow_status and 0 <= step_index < len(self.flow_status):
            self.flow_status[step_index] = "thinking"
            if showcase:
                showcase.start_step(
                    step_index=step_index,
                    agent_name=agent.name,
                    role=agent.role,
                    phase=phase or "Processing",
                    model=model,
                    emoji=agent.emoji,
                    personality=agent.personality,
                    color=agent.color,
                )
            else:
                print_flow_status(self.flow, self.flow_status, step_index)
                print_step_loading(
                    step_index=step_index,
                    total_steps=total_steps,
                    agent_name=agent.name,
                    phase=phase or "Processing",
                    color=agent.color,
                    emoji=agent.emoji,
                    personality=agent.personality,
                    model=model,
                )

        try:
            if not showcase:
                print_phase_banner(
                    agent_name=agent.name,
                    phase=phase or "Processing",
                    step_index=step_index,
                    total_steps=total_steps,
                    color=agent.color,
                    emoji=agent.emoji,
                )
                show_thinking(
                    agent.name,
                    phase=phase or "Processing",
                    step_index=step_index,
                    total_steps=total_steps,
                    emoji=agent.emoji,
                )
            else:
                time.sleep(0.25)

            if image_path and agent_key == "vision":
                user_content = self.client.build_vision_user_content(prompt=styled_prompt, image_path=image_path)
                messages = [
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [
                    {"role": "system", "content": agent.system_prompt},
                    {"role": "user", "content": styled_prompt},
                ]

            content, usage = self.client.chat_with_usage(
                model=model,
                messages=messages,
                temperature=TEMPERATURE_BY_AGENT.get(agent_key, 0.3),
                max_tokens=MAX_TOKENS_BY_AGENT.get(agent_key, 500),
            )
            content = self._polish_output(content)
            for retry_hint in (
                "Your previous output was invalid (empty or placeholder). Return a complete response that follows the template exactly.",
                "Still invalid. Return concise structured text now, and do not output 'None' or empty content.",
            ):
                if not self._is_placeholder_output(content):
                    break
                retry_messages = list(messages)
                retry_messages.append({"role": "user", "content": retry_hint})
                retry_content, retry_usage = self.client.chat_with_usage(
                    model=model,
                    messages=retry_messages,
                    temperature=TEMPERATURE_BY_AGENT.get(agent_key, 0.3),
                    max_tokens=MAX_TOKENS_BY_AGENT.get(agent_key, 500),
                )
                content = self._polish_output(retry_content)
                usage = self._merge_usage(usage, retry_usage)

            if self._is_placeholder_output(content):
                content = self._fallback_placeholder_content(agent_key=agent_key, phase=phase)
            is_error = False
        except Exception as exc:  # noqa: BLE001
            content = f"[ERROR] {exc}"
            is_error = True

        latency = time.perf_counter() - started

        if self.flow_status and 0 <= step_index < len(self.flow_status):
            self.flow_status[step_index] = "error" if is_error else "done"

        if showcase:
            showcase.stream_content(content, delay=0.01 if not is_error else 0.0)
            showcase.finish_step(
                step_index=step_index,
                is_error=is_error,
                latency=latency,
                usage=usage,
            )
            if final_highlight and not is_error:
                showcase.set_final_answer(agent_name=agent.name, content=content, emoji=agent.emoji)
        else:
            print_agent_panel(
                agent.name,
                agent.role,
                agent.emoji,
                agent.personality,
                content,
                agent.color,
                phase=phase or "Agent Response",
                step_index=step_index,
                total_steps=total_steps,
                model=model,
                is_final=final_highlight,
                is_error=is_error,
            )
            if self.flow_status and 0 <= step_index < len(self.flow_status):
                print_flow_status(self.flow, self.flow_status, step_index)
        self.history.append(
            {
                "agent_key": agent_key,
                "name": agent.name,
                "role": agent.role,
                "emoji": agent.emoji,
                "personality": agent.personality,
                "model": model,
                "phase": phase,
                "content": content,
                "error": is_error,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return content


    @staticmethod
    def _polish_output(content: str) -> str:
        text = (content or "").strip()
        if not text:
            return "(no content)"

        text = text.replace("\r\n", "\n")
        text = re.sub(r"```[\w-]*\n?", "", text)
        text = text.replace("```", "")
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"__(.*?)__", r"\1", text)
        text = re.sub(r"(?m)^\s*[-*+]\s+", "• ", text)
        text = re.sub(r"(?m)^\s*\d+\.\s+", "• ", text)
        text = re.sub(r"(?m)^\s*Question\s*:\s*.*$", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _is_placeholder_output(content: str) -> bool:
        normalized = (content or "").strip().lower().strip(".")
        return normalized in {"", "none", "null", "n/a", "na", "(no content)"}

    @staticmethod
    def _merge_usage(base: Dict[str, int], extra: Dict[str, int]) -> Dict[str, int]:
        merged: Dict[str, int] = dict(base or {})
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            merged[key] = int(merged.get(key, 0) or 0) + int(extra.get(key, 0) or 0)

        base_cost = merged.get("cost")
        extra_cost = extra.get("cost")
        if base_cost is not None or extra_cost is not None:
            try:
                merged["cost"] = float(base_cost or 0.0) + float(extra_cost or 0.0)
            except (TypeError, ValueError):
                pass
        return merged

    def _fallback_placeholder_content(self, agent_key: str, phase: str) -> str:
        previous = ""
        for entry in reversed(self.history):
            if entry.get("agent_key") == agent_key and not entry.get("error"):
                previous = entry.get("content", "").strip()
                break

        phase_lower = (phase or "").lower()
        if "rebuttal" in phase_lower:
            snippet = previous.splitlines()[0] if previous else "No earlier valid draft to reuse."
            return (
                "What Changed:\n"
                "• Fallback activated because model returned placeholder output.\n"
                "Rebuttal Response:\n"
                "• Retaining strongest prior reasoning and applying critic fixes where possible.\n"
                "Revised Answer:\n"
                f"{snippet}"
            )
        if "verdict" in phase_lower:
            return (
                "Correctness: -\n"
                "Clarity: -\n"
                "Evidence Quality: -\n"
                "Risk Flags: Model placeholder output\n"
                "Winner: Not determined\n"
                "Takeaway: Fallback verdict used because model output was invalid.\n"
                "Final Answer: Unable to produce a reliable judged answer in this run."
            )
        if "cross-exam" in phase_lower:
            return (
                "Cross-Exam Findings:\n"
                "• Fallback activated because model output was placeholder.\n"
                "Risk Flags:\n"
                "• Missing critique detail due invalid generation.\n"
                "Required Fixes:\n"
                "• Re-run this step or switch model for stronger critical analysis."
            )
        if "opening" in phase_lower:
            return (
                "Opening Position:\n"
                "• Fallback activated because model output was placeholder.\n"
                "Plan:\n"
                "• Use prior context and continue the debate without interruption.\n"
                "Assumptions:\n"
                "• Generated fallback text may be less precise than model output."
            )
        return (
            "Fallback Response:\n"
            "• Model produced placeholder output after retries.\n"
            "• Continuing pipeline with deterministic placeholder-safe content."
        )


@dataclass(frozen=True)
class AgentView:
    key: str
    name: str
    role: str
    tag: str
    emoji: str
    personality: str
    color: str
    system_prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ModelArena: terminal multi-agent LLM arena.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="arena",
        choices=["arena", "debate", "committee", "vision", "showcase"],
        help="arena (default), debate, committee, vision, or showcase",
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        help="Image path (required only in vision mode)",
    )
    parser.add_argument(
        "--question",
        dest="question",
        default=None,
        help="Run non-interactively by providing the user question directly",
    )
    parser.add_argument(
        "--theme",
        dest="theme",
        default="neon",
        choices=["neon", "sunset", "classic"],
        help="Visual theme for showcase mode",
    )
    parser.add_argument(
        "--free",
        action="store_true",
        help="Switch all roles to free model counterparts when possible",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        help="Generate a post-run research brief (hypotheses, uncertainties, next experiments)",
    )
    args = parser.parse_args()

    if args.mode == "vision" and not args.image_path:
        parser.error("vision mode requires an image path: python arena.py vision image.png")
    if args.mode != "vision" and args.image_path:
        parser.error("image_path can only be used with vision mode")
    return args


def load_runtime_config(config_path: Path) -> tuple[Dict[str, str], List[str], Dict[str, str], List[str]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    models = config.get("models")
    if not isinstance(models, dict):
        raise ValueError("config.yaml must contain a 'models' mapping.")

    missing = [key for key in REQUIRED_MODEL_KEYS if key not in models or not str(models[key]).strip()]
    if missing:
        raise ValueError(f"Missing model mappings in config.yaml: {', '.join(missing)}")

    resolved_models = {key: str(models[key]).strip() for key in REQUIRED_MODEL_KEYS}

    fallback_config = config.get("fallbacks")
    reasoner_fallback_raw = None
    if isinstance(fallback_config, dict):
        reasoner_fallback_raw = fallback_config.get("reasoner")
    if reasoner_fallback_raw is None:
        reasoner_fallback_raw = config.get("reasoner_fallbacks")

    reasoner_fallbacks: List[str] = []
    if reasoner_fallback_raw is None:
        reasoner_fallbacks = list(DEFAULT_REASONER_FALLBACK_CANDIDATES)
    elif isinstance(reasoner_fallback_raw, list):
        for item in reasoner_fallback_raw:
            value = str(item).strip()
            if value:
                reasoner_fallbacks.append(value)
    else:
        raise ValueError("config.yaml 'fallbacks.reasoner' must be a list of model IDs.")

    # Deduplicate and keep order.
    deduped: List[str] = []
    seen = set()
    for model in reasoner_fallbacks:
        if model not in seen:
            seen.add(model)
            deduped.append(model)

    free_models_raw = config.get("free_models")
    free_models = dict(DEFAULT_FREE_MODELS)
    if free_models_raw is not None:
        if not isinstance(free_models_raw, dict):
            raise ValueError("config.yaml 'free_models' must be a mapping of role -> model ID.")
        for key in REQUIRED_MODEL_KEYS:
            value = free_models_raw.get(key)
            if value is not None and str(value).strip():
                free_models[key] = str(value).strip()

    free_fallbacks_raw = config.get("free_fallbacks")
    free_reasoner_raw = None
    if isinstance(free_fallbacks_raw, dict):
        free_reasoner_raw = free_fallbacks_raw.get("reasoner")
    if free_reasoner_raw is None:
        free_reasoner_raw = config.get("free_reasoner_fallbacks")

    free_reasoner_fallbacks: List[str] = []
    if free_reasoner_raw is None:
        free_reasoner_fallbacks = list(DEFAULT_FREE_REASONER_FALLBACK_CANDIDATES)
    elif isinstance(free_reasoner_raw, list):
        for item in free_reasoner_raw:
            value = str(item).strip()
            if value:
                free_reasoner_fallbacks.append(value)
    else:
        raise ValueError("config.yaml 'free_fallbacks.reasoner' must be a list of model IDs.")

    free_seen = set()
    free_deduped: List[str] = []
    for model in free_reasoner_fallbacks:
        if model not in free_seen:
            free_seen.add(model)
            free_deduped.append(model)

    return resolved_models, deduped, free_models, free_deduped


CANONICAL_MODEL_HINTS = {
    "strategist": "hunter-alpha",
    "reasoner": "nemotron",
    "critic": "healer-alpha",
    "analyst": "gemini",
    "vision": "qwen",
}

FALLBACK_PERSONALITY_BY_ROLE = {
    "strategist": "Adaptive tactician",
    "reasoner": "Adaptive solver",
    "critic": "Adaptive challenger",
    "analyst": "Adaptive judge",
    "vision": "Adaptive visual analyst",
}


def _friendly_model_name(model_id: str) -> str:
    provider = model_id.split("/", 1)[0] if "/" in model_id else ""
    raw_name = model_id.split("/", 1)[1] if "/" in model_id else model_id
    raw_name = raw_name.split(":", 1)[0]
    chunks = [part for part in re.split(r"[-_]+", raw_name) if part]
    normalized = " ".join(chunks)
    pretty = normalized.title().replace("Vl", "VL").replace("Gpt", "GPT")
    if provider and provider not in {"openrouter"}:
        if pretty.lower().startswith(f"{provider.lower()} "):
            return pretty
        return f"{provider.title()} {pretty}"
    return pretty


def resolve_agent_view(agent_key: str, model_id: str) -> AgentView:
    base = get_agent(agent_key)
    hint = CANONICAL_MODEL_HINTS.get(agent_key, "")
    is_canonical = hint in model_id.lower() if hint else True

    if is_canonical:
        return AgentView(
            key=base.key,
            name=base.name,
            role=base.role,
            tag=base.tag,
            emoji=base.emoji,
            personality=base.personality,
            color=base.color,
            system_prompt=base.system_prompt,
        )

    return AgentView(
        key=base.key,
        name=_friendly_model_name(model_id),
        role=base.role,
        tag=base.tag,
        emoji=base.emoji,
        personality=FALLBACK_PERSONALITY_BY_ROLE.get(agent_key, base.personality),
        color=base.color,
        system_prompt=base.system_prompt,
    )


WINNER_KEYWORDS = {
    "strategist": ["hunter", "strategist"],
    "reasoner": ["nemotron", "reasoner", "deepseek"],
    "critic": ["healer", "critic"],
    "analyst": ["gemini", "analyst", "judge"],
    "vision": ["qwen", "vision"],
}


def normalize_judge_summary(summary: Dict[str, str], models: Dict[str, str]) -> Dict[str, str]:
    winner_raw = (summary.get("winner") or "").strip()
    winner_lower = winner_raw.lower()
    winner_key = ""

    # First pass: keyword matching.
    for key, keywords in WINNER_KEYWORDS.items():
        if any(token in winner_lower for token in keywords):
            winner_key = key
            break

    # Second pass: direct match against dynamic current display names.
    if not winner_key:
        for key in REQUIRED_MODEL_KEYS:
            view = resolve_agent_view(key, models.get(key, ""))
            if view.name.lower() in winner_lower:
                winner_key = key
                break

    if winner_key:
        winner_view = resolve_agent_view(winner_key, models.get(winner_key, ""))
        summary["winner"] = winner_view.name
        summary["winner_key"] = winner_key

    return summary


def is_free_model_id(model_id: str) -> bool:
    lowered = model_id.strip().lower()
    if lowered.endswith(":free"):
        return True
    return lowered in ALWAYS_FREE_MODEL_IDS


def apply_free_mode(
    models: Dict[str, str],
    reasoner_fallbacks: List[str],
    free_models: Dict[str, str],
    free_reasoner_fallbacks: List[str],
) -> tuple[Dict[str, str], List[str], List[str]]:
    updated_models = dict(models)
    notes: List[str] = []

    for key in REQUIRED_MODEL_KEYS:
        current = updated_models[key]
        if is_free_model_id(current):
            notes.append(f"{key}: already free ({current})")
            continue

        target = free_models.get(key, "").strip()
        if target:
            updated_models[key] = target
            notes.append(f"{key}: {current} -> {target}")
        else:
            notes.append(f"{key}: no free counterpart configured, keeping {current}")

    if free_reasoner_fallbacks:
        updated_reasoner_fallbacks = list(free_reasoner_fallbacks)
    else:
        updated_reasoner_fallbacks = [model for model in reasoner_fallbacks if is_free_model_id(model)]
        if not updated_reasoner_fallbacks:
            updated_reasoner_fallbacks = list(DEFAULT_FREE_REASONER_FALLBACK_CANDIDATES)

    return updated_models, updated_reasoner_fallbacks, notes


def generate_research_brief(runtime: ArenaRuntime) -> str:
    analyst_model = runtime.models["analyst"]
    analyst = resolve_agent_view("analyst", analyst_model)
    prompt = (
        "Create a concise research brief from this multi-agent transcript.\n"
        "Return plain text sections exactly:\n"
        "Working Answer:\n"
        "Confidence (0-100):\n"
        "Top Claims (max 3 bullets):\n"
        "Evidence Quality:\n"
        "Counterarguments:\n"
        "Open Questions:\n"
        "Next Experiments (max 3 bullets):\n"
        "Search Queries (max 5 bullets):\n\n"
        f"Question:\n{runtime.question}\n\n"
        f"Transcript Context:\n{summarize_entries(runtime.history)}\n\n"
        f"{OUTPUT_STYLE_GUARD}"
    )

    messages = [
        {"role": "system", "content": analyst.system_prompt},
        {"role": "user", "content": prompt},
    ]
    content, _usage = runtime.client.chat_with_usage(
        model=analyst_model,
        messages=messages,
        temperature=0.2,
        max_tokens=650,
    )
    content = runtime._polish_output(content)
    runtime.research_brief = content
    runtime.history.append(
        {
            "agent_key": "analyst",
            "name": f"{analyst.name} (Research)",
            "role": "Research Synthesizer",
            "emoji": analyst.emoji,
            "personality": analyst.personality,
            "model": analyst_model,
            "phase": "Research Brief",
            "content": content,
            "error": False,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return content


def build_display_flow(flow: List[dict], models: Dict[str, str]) -> List[dict]:
    display: List[dict] = []
    for step in flow:
        model = models.get(step["agent_key"], "-")
        agent = resolve_agent_view(step["agent_key"], model)
        display.append(
            {
                "name": agent.name,
                "role": agent.role,
                "tag": agent.tag,
                "emoji": agent.emoji,
                "personality": agent.personality,
                "color": agent.color,
                "model": model,
                "phase": step["phase"],
            }
        )
    return display


def build_all_agent_legend(models: Dict[str, str]) -> List[dict]:
    legend: List[dict] = []
    for key in REQUIRED_MODEL_KEYS:
        model = models.get(key, "-")
        agent = resolve_agent_view(key, model)
        legend.append(
            {
                "name": agent.name,
                "role": agent.role,
                "tag": agent.tag,
                "emoji": agent.emoji,
                "personality": agent.personality,
                "color": agent.color,
                "model": model,
            }
        )
    return legend


def resolve_reasoner_model(
    client: OpenRouterClient,
    configured_model: str,
    fallback_candidates: List[str],
) -> tuple[str, List[str]]:
    notes: List[str] = []
    candidates = [configured_model] + [
        model for model in fallback_candidates if model != configured_model
    ]

    for idx, model in enumerate(candidates):
        ok, error = client.probe_model(model)
        if ok:
            if idx == 0:
                notes.append(f"Reasoner probe passed: {model}")
            else:
                notes.append(f"Reasoner fallback activated: {configured_model} -> {model}")
            return model, notes
        notes.append(f"Probe failed for {model}: {error}")

    notes.append("All reasoner candidates failed probing. Keeping configured reasoner model.")
    return configured_model, notes


def _next_log_path(logs_dir: Path) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    max_idx = 0
    pattern = re.compile(r"run_(\d+)\.txt$")
    for file_path in logs_dir.glob("run_*.txt"):
        match = pattern.search(file_path.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))
    return logs_dir / f"run_{max_idx + 1:03d}.txt"


def _write_transcript(runtime: ArenaRuntime, log_path: Path) -> None:
    lines: List[str] = []
    lines.append("ModelArena Transcript")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Mode: {runtime.mode}")
    lines.append(f"Free Mode: {runtime.free_mode}")
    lines.append(f"Question: {runtime.question}")
    lines.append("Models:")
    for key in REQUIRED_MODEL_KEYS:
        lines.append(f"  {key}: {runtime.models[key]}")
    if runtime.session_notes:
        lines.append("Session Notes:")
        for note in runtime.session_notes:
            lines.append(f"  - {note}")
    lines.append("-" * 80)

    for idx, entry in enumerate(runtime.history, start=1):
        lines.append(f"[{idx}] {entry.get('emoji', '')} {entry['name']} ({entry['role']})".strip())
        if entry.get("personality"):
            lines.append(f"Persona: {entry['personality']}")
        lines.append(f"Model: {entry['model']}")
        if entry.get("phase"):
            lines.append(f"Phase: {entry['phase']}")
        lines.append(f"Time: {entry['timestamp']}")
        lines.append(entry["content"])
        lines.append("")

    log_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_arena_mode(runtime: ArenaRuntime) -> None:
    question = runtime.question
    flow = [
        {"agent_key": "strategist", "phase": "Round 1A • Opening (Strategy)"},
        {"agent_key": "reasoner", "phase": "Round 1B • Opening (Reasoning)"},
        {"agent_key": "critic", "phase": "Round 2 • Cross-Exam"},
        {"agent_key": "reasoner", "phase": "Round 3 • Rebuttal"},
        {"agent_key": "analyst", "phase": "Round 4 • Verdict"},
    ]
    runtime.start_flow(build_display_flow(flow, runtime.models))
    total_steps = len(flow)

    runtime.invoke(
        "strategist",
        (
            "Round: Opening.\n"
            "Role template (keep short):\n"
            "Opening Position:\n"
            "Plan (max 3 bullets):\n"
            "Assumptions:\n\n"
            "User question:\n"
            f"{question}\n\n"
            "Provide a clear strategic opening."
        ),
        phase=flow[0]["phase"],
        step_index=0,
        total_steps=total_steps,
    )
    runtime.invoke(
        "reasoner",
        (
            "Round: Opening.\n"
            "Role template (keep short):\n"
            "Hypothesis:\n"
            "Reasoning (max 4 bullets):\n"
            "Draft Answer:\n\n"
            "Solve the question using strategy context.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Focus on the strongest candidate answer."
        ),
        phase=flow[1]["phase"],
        step_index=1,
        total_steps=total_steps,
    )
    runtime.invoke(
        "critic",
        (
            "Round: Cross-Exam.\n"
            "Role template (keep short):\n"
            "Cross-Exam Findings (max 4 bullets):\n"
            "Risk Flags (max 3 bullets):\n"
            "Required Fixes:\n\n"
            "Critique the current reasoning.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Be direct and actionable."
        ),
        phase=flow[2]["phase"],
        step_index=2,
        total_steps=total_steps,
    )
    runtime.invoke(
        "reasoner",
        (
            "Round: Rebuttal.\n"
            "Role template (keep short):\n"
            "What Changed (max 3 bullets):\n"
            "Rebuttal Response:\n"
            "Revised Answer:\n\n"
            "Revise the solution after critique.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Address key objections and improve answer quality."
        ),
        phase=flow[3]["phase"],
        step_index=3,
        total_steps=total_steps,
    )
    verdict_text = runtime.invoke(
        "analyst",
        (
            f"{ANALYST_VERDICT_TEMPLATE}\n\n"
            "Evaluate all messages and provide verdict.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Judge the debate quality, then finalize."
        ),
        phase=flow[4]["phase"],
        step_index=4,
        total_steps=total_steps,
        final_highlight=True,
    )
    print_judge_scorecard(normalize_judge_summary(_judge_summary_from_text(verdict_text), runtime.models))
    runtime.finish_flow()


def run_debate_mode(runtime: ArenaRuntime) -> None:
    question = runtime.question
    flow = [
        {"agent_key": "reasoner", "phase": "Round 1 • Opening"},
        {"agent_key": "critic", "phase": "Round 2 • Cross-Exam"},
        {"agent_key": "reasoner", "phase": "Round 3 • Rebuttal"},
        {"agent_key": "analyst", "phase": "Round 4 • Verdict"},
    ]
    runtime.start_flow(build_display_flow(flow, runtime.models))
    total_steps = len(flow)

    runtime.invoke(
        "reasoner",
        (
            "Round: Opening.\n"
            "Role template (keep short):\n"
            "Hypothesis:\n"
            "Reasoning (max 4 bullets):\n"
            "Draft Answer:\n\n"
            "Provide an initial answer.\n"
            f"Question:\n{question}\n\n"
            "Make the opening argument strong and clear."
        ),
        phase=flow[0]["phase"],
        step_index=0,
        total_steps=total_steps,
    )
    runtime.invoke(
        "critic",
        (
            "Round: Cross-Exam.\n"
            "Role template (keep short):\n"
            "Cross-Exam Findings (max 4 bullets):\n"
            "Risk Flags (max 3 bullets):\n"
            "Required Fixes:\n\n"
            "Critique the reasoner's answer.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Focus on flaws that change the conclusion."
        ),
        phase=flow[1]["phase"],
        step_index=1,
        total_steps=total_steps,
    )
    runtime.invoke(
        "reasoner",
        (
            "Round: Rebuttal.\n"
            "Role template (keep short):\n"
            "What Changed (max 3 bullets):\n"
            "Rebuttal Response:\n"
            "Revised Answer:\n\n"
            "Revise your answer after critique.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Address the strongest objections first."
        ),
        phase=flow[2]["phase"],
        step_index=2,
        total_steps=total_steps,
    )
    verdict_text = runtime.invoke(
        "analyst",
        (
            f"{ANALYST_VERDICT_TEMPLATE}\n\n"
            "Judge the debate and provide the final answer.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Select a winner and provide a one-line takeaway."
        ),
        phase=flow[3]["phase"],
        step_index=3,
        total_steps=total_steps,
        final_highlight=True,
    )
    print_judge_scorecard(normalize_judge_summary(_judge_summary_from_text(verdict_text), runtime.models))
    runtime.finish_flow()


def run_committee_mode(runtime: ArenaRuntime) -> None:
    question = runtime.question
    flow = [
        {"agent_key": "strategist", "phase": "Round 1A • Opening (Strategy)"},
        {"agent_key": "reasoner", "phase": "Round 1B • Opening (Reasoning)"},
        {"agent_key": "vision", "phase": "Round 1C • Opening (Alt View)"},
        {"agent_key": "critic", "phase": "Round 2 • Cross-Exam"},
        {"agent_key": "reasoner", "phase": "Round 3 • Rebuttal"},
        {"agent_key": "analyst", "phase": "Round 4 • Verdict"},
    ]
    runtime.start_flow(build_display_flow(flow, runtime.models))
    total_steps = len(flow)
    runtime.invoke(
        "strategist",
        (
            "Round: Opening.\n"
            "Role template (keep short):\n"
            "Opening Position:\n"
            "Plan (max 3 bullets):\n"
            "Assumptions:\n\n"
            f"Question:\n{question}"
        ),
        phase=flow[0]["phase"],
        step_index=0,
        total_steps=total_steps,
    )
    runtime.invoke(
        "reasoner",
        (
            "Round: Opening.\n"
            "Role template (keep short):\n"
            "Hypothesis:\n"
            "Reasoning (max 4 bullets):\n"
            "Draft Answer:\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}"
        ),
        phase=flow[1]["phase"],
        step_index=1,
        total_steps=total_steps,
    )
    runtime.invoke(
        "vision",
        (
            "Round: Opening (alternative perspective).\n"
            "Role template (keep short):\n"
            "Alternative Angle:\n"
            "Supporting Signals (max 3 bullets):\n"
            "Practical Insight:\n\n"
            f"Question:\n{question}\n\n"
            "Treat this as conceptual pattern analysis when no image is provided."
        ),
        phase=flow[2]["phase"],
        step_index=2,
        total_steps=total_steps,
    )
    runtime.invoke(
        "critic",
        (
            "Round: Cross-Exam.\n"
            "Role template (keep short):\n"
            "Cross-Exam Findings (max 4 bullets):\n"
            "Risk Flags (max 3 bullets):\n"
            "Required Fixes:\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}"
        ),
        phase=flow[3]["phase"],
        step_index=3,
        total_steps=total_steps,
    )

    runtime.invoke(
        "reasoner",
        (
            "Round: Rebuttal.\n"
            "Role template (keep short):\n"
            "What Changed (max 3 bullets):\n"
            "Rebuttal Response:\n"
            "Revised Answer:\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}"
        ),
        phase=flow[4]["phase"],
        step_index=4,
        total_steps=total_steps,
    )
    verdict_text = runtime.invoke(
        "analyst",
        (
            f"{ANALYST_VERDICT_TEMPLATE}\n\n"
            "You are judging a committee-style debate. Score rigor and practicality.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}"
        ),
        phase=flow[5]["phase"],
        step_index=5,
        total_steps=total_steps,
        final_highlight=True,
    )
    print_judge_scorecard(normalize_judge_summary(_judge_summary_from_text(verdict_text), runtime.models))
    runtime.finish_flow()


def run_vision_mode(runtime: ArenaRuntime, image_path: str) -> None:
    question = runtime.question
    flow = [
        {"agent_key": "vision", "phase": "Round 1A • Opening (Visual Read)"},
        {"agent_key": "reasoner", "phase": "Round 1B • Opening (Reasoning)"},
        {"agent_key": "critic", "phase": "Round 2 • Cross-Exam"},
        {"agent_key": "reasoner", "phase": "Round 3 • Rebuttal"},
        {"agent_key": "analyst", "phase": "Round 4 • Verdict"},
    ]
    runtime.start_flow(build_display_flow(flow, runtime.models))
    total_steps = len(flow)

    runtime.invoke(
        "vision",
        (
            "Round: Opening.\n"
            "Role template (keep short):\n"
            "Visual Observations (max 4 bullets):\n"
            "Likely Implications:\n"
            "Visual Answer Draft:\n\n"
            "Describe the image and extract details relevant to this question.\n"
            f"Question:\n{question}\n\n"
            "Focus only on evidence visible in the image."
        ),
        image_path=image_path,
        phase=flow[0]["phase"],
        step_index=0,
        total_steps=total_steps,
    )
    runtime.invoke(
        "reasoner",
        (
            "Round: Opening.\n"
            "Role template (keep short):\n"
            "Hypothesis:\n"
            "Reasoning (max 4 bullets):\n"
            "Draft Answer:\n\n"
            "Solve the question using the visual analysis.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Use visual evidence explicitly."
        ),
        phase=flow[1]["phase"],
        step_index=1,
        total_steps=total_steps,
    )
    runtime.invoke(
        "critic",
        (
            "Round: Cross-Exam.\n"
            "Role template (keep short):\n"
            "Cross-Exam Findings (max 4 bullets):\n"
            "Risk Flags (max 3 bullets):\n"
            "Required Fixes:\n\n"
            "Critique the current solution and visual interpretation.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Call out unsupported claims."
        ),
        phase=flow[2]["phase"],
        step_index=2,
        total_steps=total_steps,
    )
    runtime.invoke(
        "reasoner",
        (
            "Round: Rebuttal.\n"
            "Role template (keep short):\n"
            "What Changed (max 3 bullets):\n"
            "Rebuttal Response:\n"
            "Revised Answer:\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}"
        ),
        phase=flow[3]["phase"],
        step_index=3,
        total_steps=total_steps,
    )
    verdict_text = runtime.invoke(
        "analyst",
        (
            f"{ANALYST_VERDICT_TEMPLATE}\n\n"
            "Provide the final judged answer for a vision-grounded debate.\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{summarize_entries(runtime.history)}\n\n"
            "Require strong evidence grounding."
        ),
        phase=flow[4]["phase"],
        step_index=4,
        total_steps=total_steps,
        final_highlight=True,
    )
    print_judge_scorecard(normalize_judge_summary(_judge_summary_from_text(verdict_text), runtime.models))
    runtime.finish_flow()


def run_showcase_mode(runtime: ArenaRuntime, theme: str) -> None:
    question = runtime.question
    flow = [
        {"agent_key": "strategist", "phase": "Round 1A • Opening (Strategy)"},
        {"agent_key": "reasoner", "phase": "Round 1B • Opening (Reasoning)"},
        {"agent_key": "critic", "phase": "Round 2 • Cross-Exam"},
        {"agent_key": "reasoner", "phase": "Round 3 • Rebuttal"},
        {"agent_key": "analyst", "phase": "Round 4 • Verdict"},
    ]
    display_flow = build_display_flow(flow, runtime.models)
    runtime.start_flow(display_flow, render=False)
    total_steps = len(flow)

    def step_prompt(idx: int) -> tuple[str, str]:
        context = summarize_entries(runtime.history)
        if idx == 0:
            return (
                "strategist",
                (
                    "Round: Opening.\n"
                    "Role template (keep short):\n"
                    "Opening Position:\n"
                    "Plan (max 3 bullets):\n"
                    "Assumptions:\n\n"
                    f"Question:\n{question}"
                ),
            )
        if idx == 1:
            return (
                "reasoner",
                (
                    "Round: Opening.\n"
                    "Role template (keep short):\n"
                    "Hypothesis:\n"
                    "Reasoning (max 4 bullets):\n"
                    "Draft Answer:\n\n"
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}"
                ),
            )
        if idx == 2:
            return (
                "critic",
                (
                    "Round: Cross-Exam.\n"
                    "Role template (keep short):\n"
                    "Cross-Exam Findings (max 4 bullets):\n"
                    "Risk Flags (max 3 bullets):\n"
                    "Required Fixes:\n\n"
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}"
                ),
            )
        if idx == 3:
            return (
                "reasoner",
                (
                    "Round: Rebuttal.\n"
                    "Role template (keep short):\n"
                    "What Changed (max 3 bullets):\n"
                    "Rebuttal Response:\n"
                    "Revised Answer:\n\n"
                    f"Question:\n{question}\n\n"
                    f"Context:\n{context}"
                ),
            )
        return (
            "analyst",
            (
                f"{ANALYST_VERDICT_TEMPLATE}\n\n"
                "Judge the full debate and provide final verdict.\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}"
            ),
        )

    if not supports_live_layout():
        print_warning(
            "Showcase live layout needs a wider terminal. Falling back to stable linear showcase view "
            "(resize wider to re-enable full live board)."
        )
        verdict_text = ""
        for idx, step in enumerate(flow):
            agent_key, prompt = step_prompt(idx)
            output = runtime.invoke(
                agent_key,
                prompt,
                phase=step["phase"],
                step_index=idx,
                total_steps=total_steps,
                final_highlight=(idx == total_steps - 1),
            )
            if idx == total_steps - 1:
                verdict_text = output
        print_judge_scorecard(normalize_judge_summary(_judge_summary_from_text(verdict_text), runtime.models))
        runtime.finish_flow(render=True)
        return

    with ShowcaseSession(mode="showcase", question=question, flow=display_flow, theme=theme) as board:
        verdict_text = ""
        for idx, step in enumerate(flow):
            agent_key, prompt = step_prompt(idx)
            output = runtime.invoke(
                agent_key,
                prompt,
                phase=step["phase"],
                step_index=idx,
                total_steps=total_steps,
                final_highlight=(idx == total_steps - 1),
                showcase=board,
            )
            if idx == total_steps - 1:
                verdict_text = output
    print_judge_scorecard(normalize_judge_summary(_judge_summary_from_text(verdict_text), runtime.models))
    runtime.finish_flow(render=False)


def main() -> int:
    args = parse_args()
    print_header(args.mode)

    question = args.question.strip() if args.question else prompt_question()
    if not question:
        print_warning("Question cannot be empty.")
        return 1

    try:
        models, reasoner_fallbacks, free_models, free_reasoner_fallbacks = load_runtime_config(CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        print_warning(str(exc))
        return 1

    try:
        client = OpenRouterClient()
    except Exception as exc:  # noqa: BLE001
        print_warning(str(exc))
        return 1

    free_notes: List[str] = []
    if args.free:
        models, reasoner_fallbacks, free_notes = apply_free_mode(
            models=models,
            reasoner_fallbacks=reasoner_fallbacks,
            free_models=free_models,
            free_reasoner_fallbacks=free_reasoner_fallbacks,
        )

    resolved_reasoner, reasoner_notes = resolve_reasoner_model(
        client,
        models["reasoner"],
        reasoner_fallbacks,
    )
    models["reasoner"] = resolved_reasoner

    print_mission(question)
    print_agent_legend(build_all_agent_legend(models))
    if args.free:
        print_info("Free Mode Enabled: switched roles to free counterparts where configured.")
        for note in free_notes:
            print_info(note)
    for note in reasoner_notes:
        if note.startswith("Probe failed") and not (
            os.getenv("MODELARENA_DEBUG_PROBE", "0") == "1"
        ):
            continue
        if note.startswith("Reasoner fallback activated"):
            print_warning(note)
        else:
            print_info(note)

    runtime = ArenaRuntime(
        client=client,
        models=models,
        mode=args.mode,
        question=question,
        free_mode=args.free,
    )
    runtime.session_notes.extend(free_notes)
    runtime.session_notes.extend(reasoner_notes)

    if args.mode == "arena":
        run_arena_mode(runtime)
    elif args.mode == "debate":
        run_debate_mode(runtime)
    elif args.mode == "committee":
        run_committee_mode(runtime)
    elif args.mode == "vision":
        run_vision_mode(runtime, args.image_path)
    elif args.mode == "showcase":
        run_showcase_mode(runtime, args.theme)
    else:
        print_warning(f"Unknown mode: {args.mode}")
        return 1

    if args.research:
        try:
            print_info("Generating research brief...")
            brief = generate_research_brief(runtime)
            print_research_brief(brief, runtime.models["analyst"])
        except Exception as exc:  # noqa: BLE001
            print_warning(f"Research brief failed: {exc}")

    log_path = _next_log_path(LOGS_DIR)
    _write_transcript(runtime, log_path)
    print_info(f"Transcript saved to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
