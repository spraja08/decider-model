"""Load the records used by the MiniJev notebook."""
from __future__ import annotations

from pathlib import Path
from typing import Any

OPTION_COUNT = 4
PROMPT_CHARS = set("abcdefghijklmnopqrstuvwxyzQOA:;| ")
TEXT_CHARS = set("abcdefghijklmnopqrstuvwxyz ")


def validate_prompt_text(
    value: str, source_path: str | Path, line_number: int, field: str
) -> None:
    """Validate lowercase text embedded in a prompt."""
    if (
        not value
        or value != value.strip()
        or "  " in value
        or set(value) - TEXT_CHARS
        or any(delimiter in value for delimiter in ("|", ";", ":"))
    ):
        raise ValueError(f"{source_path}:{line_number}: invalid spaced prompt {field}")


def parse_prompt(
    prompt: str, source_path: str | Path = "<prompt>", line_number: int = 1
) -> tuple[str, list[str], list[int], int]:
    """Parse a prompt and return its question, options, and structural positions."""
    if (
        not prompt.startswith("Q:")
        or prompt.count("Q:") != 1
        or prompt.count("|") != 1
        or prompt.count("O:") != OPTION_COUNT
        or not prompt.endswith("A")
        or set(prompt) - PROMPT_CHARS
    ):
        raise ValueError(f"{source_path}:{line_number}: malformed prompt")
    context, suffix = prompt.split("|")
    if len(context) <= 2:
        raise ValueError(f"{source_path}:{line_number}: empty question or metadata")
    validate_prompt_text(context[2:], source_path, line_number, "question")
    parts = suffix[:-1].split(";")
    if len(parts) != OPTION_COUNT + 1 or parts[-1] != "":
        raise ValueError(
            f"{source_path}:{line_number}: options must end in four semicolons"
        )
    options: list[str] = []
    option_positions: list[int] = []
    cursor = len(context) + 1
    for part in parts[:-1]:
        if not part.startswith("O:") or len(part) <= 2:
            raise ValueError(f"{source_path}:{line_number}: malformed option")
        option = part[2:]
        validate_prompt_text(option, source_path, line_number, "option")
        options.append(option)
        cursor += len(part)
        option_positions.append(cursor)
        cursor += 1
    return context[2:], options, option_positions, len(prompt) - 1


def parse_record(
    line: str, source_path: str | Path = "<record>", line_number: int = 1
) -> dict[str, Any]:
    """Parse one four-field TSV record while retaining model position metadata."""
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 4:
        raise ValueError(
            f"{source_path}:{line_number}: expected prompt, label, domain, and fact_id"
        )
    prompt, label_text, domain, fact_id = fields
    try:
        label = int(label_text)
    except ValueError as error:
        raise ValueError(f"{source_path}:{line_number}: non-integer label") from error
    if not domain or not fact_id:
        raise ValueError(f"{source_path}:{line_number}: empty question or metadata")
    question, options, option_positions, answer_position = parse_prompt(
        prompt, source_path, line_number
    )
    if len(options) != len(set(options)) or not 0 <= label < OPTION_COUNT:
        raise ValueError(
            f"{source_path}:{line_number}: expected four unique options and label 0..3"
        )
    return {
        "prompt": prompt,
        "question": question,
        "options": options,
        "label": label,
        "domain": domain,
        "fact_id": fact_id,
        "option_positions": option_positions,
        "answer_position": answer_position,
    }


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty TSV lines and include the source path in parse errors."""
    path = Path(path)
    return [
        parse_record(line, path, number)
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if line
    ]
