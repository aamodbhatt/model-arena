"""Rich-based terminal UI helpers for ModelArena."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


console = Console()
COMPACT_WIDTH = 128
MIN_SHOWCASE_WIDTH = 140
THEME_STYLES: Dict[str, Dict[str, str]] = {
    "neon": {"accent": "bright_cyan", "line": "bright_blue", "ok": "green", "warn": "yellow", "bad": "red"},
    "sunset": {"accent": "bright_magenta", "line": "magenta", "ok": "green", "warn": "yellow", "bad": "red"},
    "classic": {"accent": "white", "line": "blue", "ok": "green", "warn": "yellow", "bad": "red"},
}


class ShowcaseSession:
    def __init__(self, mode: str, question: str, flow: List[dict], theme: str = "neon") -> None:
        self.mode = mode
        self.question = question
        self.flow = flow
        self.theme = theme if theme in THEME_STYLES else "neon"
        self.styles = THEME_STYLES[self.theme]
        self.show_emoji = emojis_enabled()

        self.statuses = ["pending"] * len(flow)
        self.current_step = 0
        self.active_emoji = "🤖"
        self.active_agent = "Waiting"
        self.active_role = ""
        self.active_personality = ""
        self.active_color = "white"
        self.active_phase = "Booting Arena"
        self.active_model = "-"
        self.current_output = "Arena initializing..."
        self.final_answer = ""
        self.final_emoji = ""
        self.final_by = ""
        self.recent_events: List[str] = []

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.error_count = 0
        self.start_time = time.time()
        self.last_latency = 0.0

        self._live: Optional[Live] = None

    def __enter__(self) -> "ShowcaseSession":
        self._live = Live(self._render(), console=console, refresh_per_second=12)
        self._live.__enter__()
        self._refresh()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._live:
            self._live.__exit__(exc_type, exc, tb)
            self._live = None

    def start_step(
        self,
        step_index: int,
        agent_name: str,
        role: str,
        phase: str,
        model: str,
        emoji: str = "🤖",
        personality: str = "",
        color: str = "white",
    ) -> None:
        self.current_step = step_index
        self.statuses[step_index] = "thinking"
        self.active_emoji = emoji
        self.active_agent = agent_name
        self.active_role = role
        self.active_personality = personality
        self.active_color = color or "white"
        self.active_phase = phase
        self.active_model = model
        self.current_output = ""
        self._add_event(f"Step {step_index + 1} started: {agent_name} - {phase}")
        self._refresh()

    def stream_content(self, content: str, chunk_size: int = 48, delay: float = 0.018) -> None:
        if not content:
            self.current_output = "(no content)"
            self._refresh()
            return

        self.current_output = ""
        for idx in range(0, len(content), chunk_size):
            self.current_output += content[idx : idx + chunk_size]
            self._refresh()
            time.sleep(delay)

    def finish_step(self, step_index: int, is_error: bool, latency: float, usage: Dict[str, Any]) -> None:
        self.statuses[step_index] = "error" if is_error else "done"
        self.last_latency = latency
        if is_error:
            self.error_count += 1

        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        cost = usage.get("cost")
        if cost is not None:
            try:
                self.total_cost += float(cost)
            except (TypeError, ValueError):
                pass

        state = "ERROR" if is_error else "DONE"
        self._add_event(
            f"Step {step_index + 1} {state} in {latency:.2f}s "
            f"(in:{prompt_tokens} out:{completion_tokens})"
        )
        self._refresh()

    def set_final_answer(self, agent_name: str, content: str, emoji: str = "") -> None:
        self.final_emoji = emoji
        self.final_by = agent_name
        self.final_answer = content.strip() or "(no content)"
        self._add_event(f"Final verdict produced by {agent_name}")
        self._refresh()

    def note_failover(
        self,
        agent_name: str,
        role: str,
        personality: str,
        color: str,
        old_model: str,
        new_model: str,
    ) -> None:
        self.active_agent = agent_name
        self.active_role = role
        self.active_personality = personality
        self.active_color = color or "white"
        self.active_model = new_model
        self._add_event(f"Failover: {old_model} -> {new_model}")
        self._refresh()

    def _add_event(self, event: str) -> None:
        self.recent_events.append(event)
        self.recent_events = self.recent_events[-6:]

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._render())

    def _render(self) -> Layout:
        root = Layout()
        root.split_column(
            Layout(name="top", size=3),
            Layout(name="body"),
            Layout(name="bottom", size=9),
        )
        root["body"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="center", ratio=6),
            Layout(name="right", ratio=3),
        )

        elapsed = time.time() - self.start_time
        top_title = (
            f"SHOWCASE MODE • {self.mode.upper()} • theme={self.theme} • "
            f"elapsed={elapsed:.1f}s • active={self.active_agent}"
        )
        root["top"].update(
            Panel(top_title, border_style=self.styles["line"], title="ModelArena Live", title_align="left")
        )
        root["left"].update(self._render_timeline())
        root["center"].update(self._render_live_feed())
        root["right"].update(self._render_metrics())
        root["bottom"].update(self._render_bottom())
        return root

    def _render_timeline(self) -> Panel:
        table = Table(show_header=True, header_style=f"bold {self.styles['accent']}", box=None, expand=True)
        table.add_column("Step", width=4)
        table.add_column("Agent", min_width=10, no_wrap=False, overflow="fold")
        table.add_column("Persona", min_width=10, no_wrap=False, overflow="fold")
        table.add_column("Status", width=8)
        for idx, item in enumerate(self.flow):
            status = self.statuses[idx]
            icon = {
                "pending": "[dim]○[/dim]",
                "thinking": f"[bold {self.styles['warn']}]◔[/bold {self.styles['warn']}]",
                "done": f"[bold {self.styles['ok']}]●[/bold {self.styles['ok']}]",
                "error": f"[bold {self.styles['bad']}]✖[/bold {self.styles['bad']}]",
            }.get(status, "[dim]?[/dim]")
            emoji = item.get("emoji", "")
            persona = item.get("personality", "")
            color = item.get("color", "white")
            tag = item.get("tag", item["name"].split()[0].upper())
            prefix = emoji if self.show_emoji else tag
            agent_label = f"[bold {color}]{prefix} {item['name']}[/bold {color}]".strip()
            table.add_row(str(idx + 1), agent_label, persona, icon)
        return Panel(table, title="Flow", title_align="left", border_style=self.styles["line"])

    def _render_live_feed(self) -> Panel:
        agent_prefix = self.active_emoji if self.show_emoji else "AGENT"
        title = (
            f"[bold {self.active_color}]"
            f"{agent_prefix} {self.active_agent} ({self.active_role})"
            f"[/bold {self.active_color}]"
        )
        subtitle = (
            f"Phase: {self.active_phase}  |  Persona: {self.active_personality or 'Adaptive'}"
            f"  |  Model: {self.active_model}"
        )
        body = Text(self.current_output or "waiting for output...", no_wrap=False, overflow="fold")
        return Panel(body, title=title, subtitle=subtitle, title_align="left", border_style=self.styles["accent"])

    def _render_metrics(self) -> Panel:
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("k", style="bold")
        table.add_column("v")
        table.add_row("Errors", str(self.error_count))
        table.add_row("Last Latency", f"{self.last_latency:.2f}s")
        table.add_row("Prompt Tokens", str(self.total_prompt_tokens))
        table.add_row("Completion Tokens", str(self.total_completion_tokens))
        table.add_row("Cost", f"${self.total_cost:.6f}")
        events = "\n".join(f"• {event}" for event in self.recent_events) or "• waiting"
        block = Table(show_header=False, box=None, expand=True)
        block.add_row(table)
        block.add_row(Text(""))
        block.add_row(Text("Events", style="bold"))
        block.add_row(Text(events))
        return Panel(block, title="Telemetry", title_align="left", border_style=self.styles["line"])

    def _render_bottom(self) -> Panel:
        if self.final_answer:
            body = Text(self.final_answer, no_wrap=False, overflow="fold")
            subtitle = f"by {self.final_emoji} {self.final_by}".strip()
        else:
            body = Text(self.question, no_wrap=False, overflow="fold")
            subtitle = "Awaiting final verdict"
        return Panel(body, title="Final Card", subtitle=subtitle, title_align="left", border_style="bright_green")


def print_header(mode: str) -> None:
    console.print(Rule("[bold bright_cyan]ModelArena[/bold bright_cyan]", style="bright_blue"))
    console.print(
        f"[bold bright_blue]Mode:[/bold bright_blue] {mode}  [dim]|  Live Multi-Agent Debate[/dim]",
    )


def prompt_question() -> str:
    return console.input("\n[bold white]Enter your question:[/bold white] ").strip()


def print_mission(question: str) -> None:
    panel = Panel(
        Text(question),
        title="Mission Brief",
        title_align="left",
        border_style="bright_blue",
    )
    console.print(panel)


def ui_width() -> int:
    try:
        return int(console.size.width)
    except Exception:  # noqa: BLE001
        return 120


def is_compact_ui() -> bool:
    return ui_width() < COMPACT_WIDTH


def emojis_enabled() -> bool:
    if os.getenv("MODELARENA_EMOJI", "1") == "0":
        return False
    encoding = (console.encoding or "").lower()
    return "utf" in encoding


def supports_live_layout() -> bool:
    term = (os.getenv("TERM") or "").lower()
    if term in {"", "dumb"}:
        return False
    return ui_width() >= MIN_SHOWCASE_WIDTH


def _agent_token(item: dict) -> str:
    color = item.get("color", "white")
    tag = item.get("tag", item.get("name", "AGENT").split()[0].upper())
    if emojis_enabled():
        token = item.get("emoji", tag)
    else:
        token = tag
    return f"[bold {color}]{token}[/bold {color}]"


def print_agent_legend(agents: List[dict]) -> None:
    table = Table(show_header=True, header_style="bold white", box=None, expand=True, padding=(0, 1))
    table.add_column("Agent", min_width=16, no_wrap=False, overflow="fold")
    table.add_column("Persona", min_width=14, no_wrap=False, overflow="fold")
    table.add_column("Model", min_width=20, no_wrap=False, overflow="fold")
    if not is_compact_ui():
        table.add_column("Color", width=10)

    for item in agents:
        color = item.get("color", "white")
        prefix = _agent_token(item)
        if is_compact_ui():
            label = f"{prefix} [bold {color}]{item.get('name', '')}[/bold {color}]"
        else:
            label = f"{prefix} [bold {color}]{item.get('name', '')} ({item.get('role', '')})[/bold {color}]"
        model = item.get("model", "-")
        if is_compact_ui():
            table.add_row(label, item.get("personality", ""), model)
        else:
            table.add_row(label, item.get("personality", ""), model, f"[{color}]{color}[/{color}]")

    console.print(Panel(table, title="Agent Personalities", title_align="left", border_style="bright_blue"))


def print_flow_plan(flow: List[dict]) -> None:
    table = Table(show_header=True, header_style="bold white", box=None, expand=True, padding=(0, 1))
    table.add_column("Step", width=4)
    table.add_column("Agent", min_width=12, no_wrap=False, overflow="fold")
    table.add_column("Model", min_width=20, no_wrap=False, overflow="fold")
    if not is_compact_ui():
        table.add_column("Persona", min_width=12, no_wrap=False, overflow="fold")
    table.add_column("Phase", min_width=16, no_wrap=False, overflow="fold")
    table.add_column("Status", width=10)

    for idx, item in enumerate(flow, start=1):
        color = item.get("color", "white")
        token = _agent_token(item)
        name = item.get("name", "")
        role = item.get("role", "")
        model = item.get("model", "-")
        phase = item.get("phase", "")
        if is_compact_ui():
            agent_label = f"{token} [bold {color}]{name}[/bold {color}]"
        else:
            agent_label = f"{token} [bold {color}]{name} ({role})[/bold {color}]"
        if is_compact_ui():
            table.add_row(str(idx), agent_label, model, phase, "[dim]PENDING[/dim]")
        else:
            table.add_row(
                str(idx),
                agent_label,
                model,
                item.get("personality", ""),
                phase,
                "[dim]PENDING[/dim]",
            )

    console.print(Panel(table, title="Arena Flow", title_align="left", border_style="bright_blue"))


def print_flow_status(flow: List[dict], statuses: List[str], step_index: int) -> None:
    if is_compact_ui():
        table = Table(show_header=True, header_style="bold white", box=None, expand=True)
        table.add_column("Step", width=4)
        table.add_column("Agent", min_width=10, no_wrap=False, overflow="fold")
        table.add_column("State", width=10)
        for idx, item in enumerate(flow):
            status = statuses[idx]
            state = {
                "pending": "[dim]PENDING[/dim]",
                "thinking": "[bold yellow]THINKING[/bold yellow]",
                "done": "[bold green]DONE[/bold green]",
                "error": "[bold red]ERROR[/bold red]",
            }.get(status, "[dim]?[/dim]")
            color = item.get("color", "white")
            label = item.get("tag", item.get("name", "AGENT").split()[0].upper())
            if idx == step_index:
                label = f"[bold bright_white]{label}[/bold bright_white]"
            table.add_row(str(idx + 1), f"[{color}]{label}[/{color}]", state)
        console.print(Panel(table, title="Arena Progress", title_align="left", border_style="blue"))
        return

    parts: List[str] = []
    for idx, item in enumerate(flow):
        status = statuses[idx]
        icon = {
            "pending": "[dim]○[/dim]",
            "thinking": "[bold yellow]◔[/bold yellow]",
            "done": "[bold green]●[/bold green]",
            "error": "[bold red]✖[/bold red]",
        }.get(status, "[dim]?[/dim]")
        short_name = item.get("tag", item["name"].split()[0].upper())
        color = item.get("color", "white")
        chip = f"{idx + 1}.[{color}]{short_name}[/{color}] {icon}".strip()
        if idx == step_index:
            chip = f"[bold bright_white]{chip}[/bold bright_white]"
        parts.append(chip)

    summary = "  [dim]→[/dim]  ".join(parts)
    console.print(Panel(summary, title="Arena Progress", title_align="left", border_style="blue"))


def show_thinking(
    agent_name: str,
    phase: str,
    step_index: int,
    total_steps: int,
    delay_seconds: float = 1.0,
    emoji: str = "",
) -> None:
    safe_emoji = emoji if emojis_enabled() else ""
    if os.getenv("MODELARENA_NO_SPINNER", "0") == "1":
        console.print(
            f"[yellow]...[/yellow] Step {step_index + 1}/{total_steps} "
            f"{safe_emoji} {agent_name} :: {phase}".strip()
        )
        time.sleep(max(delay_seconds, 0.2))
        return
    with console.status(
        f"[bold yellow]Step {step_index + 1}/{total_steps}[/bold yellow] "
        f"{safe_emoji} {agent_name} :: {phase}".strip(),
        spinner="dots",
    ):
        time.sleep(max(delay_seconds, 0.2))


def print_step_loading(
    step_index: int,
    total_steps: int,
    agent_name: str,
    phase: str,
    color: str,
    emoji: str = "",
    personality: str = "",
    model: str = "",
) -> None:
    prefix = emoji if emojis_enabled() else "..."
    body = (
        f"{prefix} Preparing next move...\n\n"
        f"Step: {step_index + 1}/{total_steps}\n"
        f"Agent: {agent_name}\n"
        f"Model: {model or '-'}\n"
        f"Phase: {phase}\n"
        f"Persona: {personality or 'Adaptive'}"
    )
    console.print(
        Panel(
            Text(body),
            title="Thinking",
            title_align="left",
            border_style=color,
            expand=True,
        )
    )


def print_phase_banner(
    agent_name: str,
    phase: str,
    step_index: int,
    total_steps: int,
    color: str,
    emoji: str = "",
) -> None:
    token = emoji if emojis_enabled() else "AGENT"
    console.print(
        Rule(
            f"[bold {color}]Step {step_index + 1}/{total_steps}[/bold {color}] • "
            f"[bold]{token} {agent_name}[/bold] • {phase}".strip(),
            style=color,
        )
    )


def print_agent_panel(
    name: str,
    role: str,
    emoji: str,
    personality: str,
    content: str,
    color: str,
    phase: str,
    step_index: int,
    total_steps: int,
    model: str = "",
    is_final: bool = False,
    is_error: bool = False,
) -> None:
    if is_error:
        body = Text(content.strip() or "(no content)", style="bold red")
    else:
        body = Text(content.strip() or "(no content)", no_wrap=False, overflow="fold")

    token = emoji if emojis_enabled() else role.split()[0].upper()
    if is_compact_ui():
        title = f"Step {step_index + 1}/{total_steps} • {token} {name}".strip()
    else:
        title = f"Step {step_index + 1}/{total_steps} • {token} {name} ({role})".strip()
    title = f"[bold {color}]{title}[/bold {color}]"
    if is_compact_ui():
        subtitle = f"{phase}  |  {model or '-'}"
    else:
        subtitle = f"{phase}  |  {personality}  |  {model or '-'}".strip(" |")
    panel = Panel(
        body,
        title=title,
        title_align="left",
        subtitle=subtitle,
        subtitle_align="right",
        border_style=color,
        expand=True,
    )
    console.print(panel)
    if is_final and not is_error:
        print_final_answer(name, content, emoji=token)


def print_final_answer(agent_name: str, content: str, emoji: str = "") -> None:
    extracted = _extract_final_answer(content)
    final_panel = Panel(
        Text(extracted, no_wrap=False, overflow="fold"),
        title="Final Answer",
        subtitle=f"by {emoji} {agent_name}".strip(),
        title_align="left",
        border_style="bright_green",
        expand=True,
    )
    console.print(final_panel)


def _extract_final_answer(content: str) -> str:
    text = content.strip()
    if not text:
        return "(no content)"

    match = re.search(
        r"(?is)(?:^|\n)(?:final answer|final recommendation|decision|verdict)\s*[:\-]\s*(.+?)(?:\n[A-Z][A-Za-z ]{2,30}\s*:\s|\Z)",
        text,
    )
    extracted = match.group(1).strip() if match else text
    if len(extracted) > 1400:
        extracted = extracted[:1397] + "..."
    return extracted


def print_info(message: str) -> None:
    console.print(Text(message, style="bold bright_blue"))


def print_warning(message: str) -> None:
    panel = Panel(Text(message), title="Warning", title_align="left", border_style="red")
    console.print(panel)


def print_research_brief(content: str, model: str) -> None:
    panel = Panel(
        Text(content.strip() or "(no research brief generated)", no_wrap=False, overflow="fold"),
        title="Research Brief",
        subtitle=f"by {model}",
        title_align="left",
        border_style="bright_magenta",
        expand=True,
    )
    console.print(panel)


def print_judge_scorecard(summary: Dict[str, str]) -> None:
    winner = summary.get("winner", "Not specified")
    takeaway = summary.get("takeaway", "No takeaway provided.")
    correctness = summary.get("correctness", "-")
    clarity = summary.get("clarity", "-")
    evidence = summary.get("evidence_quality", "-")
    risk_flags = summary.get("risk_flags", "None listed")

    winner_color = "bright_green"
    winner_token = "WINNER"
    winner_by_key = {
        "strategist": ("cyan", "🧭", "STRAT"),
        "reasoner": ("green", "🧠", "REASON"),
        "critic": ("red", "🛡️", "CRITIC"),
        "analyst": ("yellow", "⚖️", "JUDGE"),
        "vision": ("magenta", "👁️", "VISION"),
    }
    winner_key = (summary.get("winner_key") or "").strip().lower()
    if winner_key in winner_by_key:
        color, icon, tag = winner_by_key[winner_key]
        winner_color = color
        winner_token = icon if emojis_enabled() else tag
    else:
        winner_map = {
            "hunter": ("cyan", "🧭", "STRAT"),
            "nemotron": ("green", "🧠", "REASON"),
            "deepseek": ("green", "🧠", "REASON"),
            "healer": ("red", "🛡️", "CRITIC"),
            "gemini": ("yellow", "⚖️", "JUDGE"),
            "qwen": ("magenta", "👁️", "VISION"),
        }
        lowered = winner.lower()
        for key, (color, icon, tag) in winner_map.items():
            if key in lowered:
                winner_color = color
                winner_token = icon if emojis_enabled() else tag
                break

    def parse_score(raw: str) -> float | None:
        match = re.search(r"(\d+(?:\.\d+)?)", raw or "")
        if not match:
            return None
        value = float(match.group(1))
        return max(0.0, min(10.0, value))

    def meter(score: float | None) -> str:
        if score is None:
            return "-"
        filled = int(round(score))
        empty = max(0, 10 - filled)
        tone = "green" if score >= 8 else "yellow" if score >= 5 else "red"
        return f"[{tone}]{'#' * filled}{'-' * empty}[/{tone}]"

    rows = [
        ("Correctness", correctness),
        ("Clarity", clarity),
        ("Evidence Quality", evidence),
        ("Risk Flags", risk_flags),
    ]

    rubric = Table(show_header=True, header_style="bold white", box=None, expand=True, padding=(0, 1))
    rubric.add_column("Rubric", min_width=16, no_wrap=False, overflow="fold")
    if is_compact_ui():
        rubric.add_column("Score/Detail", no_wrap=False, overflow="fold")
        for label, value in rows:
            rubric.add_row(label, value)
    else:
        rubric.add_column("Value", width=10)
        rubric.add_column("Meter", width=14)
        rubric.add_column("Detail", no_wrap=False, overflow="fold")
        for label, value in rows:
            if label == "Risk Flags":
                rubric.add_row(label, "-", "-", value)
            else:
                score = parse_score(value)
                rubric.add_row(label, value, meter(score), "")

    winner_text = (
        f"[bold {winner_color}][WINNER][/bold {winner_color}] "
        f"[bold {winner_color}]{winner_token} {winner}[/bold {winner_color}]\n"
        f"Takeaway: {takeaway}"
    )
    badge = Panel(
        Text.from_markup(winner_text),
        title="Verdict Summary",
        title_align="left",
        border_style=winner_color,
        expand=True,
    )
    console.print(badge)
    console.print(Panel(rubric, title="Judge Rubric", title_align="left", border_style="yellow"))

    final_answer = (summary.get("final_answer") or "").strip()
    if final_answer:
        console.print(
            Panel(
                Text(final_answer, no_wrap=False, overflow="fold"),
                title="Judge Final Answer",
                title_align="left",
                border_style="bright_green",
                expand=True,
            )
        )
