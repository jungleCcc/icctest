from __future__ import annotations

from pathlib import Path
import subprocess

from . import INK_CHANNELS


class CmmError(RuntimeError):
    pass


def _build_xicclu_input(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for row in rows:
        values = []
        for channel in INK_CHANNELS:
            value = float(row[channel])
            if value < 0 or value > 100:
                raise CmmError(f"{channel} is outside 0-100 for sample {row.get('SAMPLE_NAME')}: {value}")
            values.append(f"{value:.6f}")
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


def _parse_lab_output(output: str, expected_count: int) -> list[dict[str, float]]:
    labs: list[dict[str, float]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            raise CmmError(f"Could not parse Lab output line: {line}")
        try:
            lab_l, lab_a, lab_b = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise CmmError(f"Could not parse Lab output line: {line}") from exc
        labs.append({"Lab_L": lab_l, "Lab_a": lab_a, "Lab_b": lab_b})

    if len(labs) != expected_count:
        raise CmmError(f"Expected {expected_count} Lab rows from xicclu, got {len(labs)}")
    return labs


def convert_7clr_to_lab(
    rows: list[dict[str, object]],
    icc_path: str | Path,
    xicclu_path: str = "xicclu",
) -> list[dict[str, object]]:
    profile = Path(icc_path)
    if not profile.exists():
        raise FileNotFoundError(f"ICC profile not found: {profile}")
    if not rows:
        return []

    command = [
        xicclu_path,
        "-v0",
        "-ff",
        "-ir",
        "-pl",
        "-s",
        "100",
        str(profile),
    ]
    input_text = _build_xicclu_input(rows)

    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise CmmError(
            "xicclu executable not found. Install ArgyllCMS and ensure xicclu is on PATH, "
            "or pass --xicclu with the executable path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise CmmError(f"xicclu conversion failed: {stderr}") from exc

    labs = _parse_lab_output(result.stdout, expected_count=len(rows))
    converted: list[dict[str, object]] = []
    for row, lab in zip(rows, labs):
        merged = dict(row)
        merged.update(lab)
        converted.append(merged)
    return converted
