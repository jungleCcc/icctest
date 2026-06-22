from __future__ import annotations

from pathlib import Path
import subprocess

from . import INK_CHANNELS


class CmmError(RuntimeError):
    pass


def _build_7clr_input(rows: list[dict[str, object]]) -> str:
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


def _build_xicclu_input(rows: list[dict[str, object]]) -> str:
    return _build_7clr_input(rows)


def _parse_lab_output(output: str, expected_count: int) -> list[dict[str, float]]:
    labs: list[dict[str, float]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parts = stripped.split()
            if len(parts) < 3:
                continue
            lab_l, lab_a, lab_b = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            continue
        labs.append({"Lab_L": lab_l, "Lab_a": lab_a, "Lab_b": lab_b})

    if len(labs) != expected_count:
        raise CmmError(f"Expected {expected_count} Lab rows from CMM output, got {len(labs)}")
    return labs


def convert_7clr_to_lab(
    rows: list[dict[str, object]],
    icc_path: str | Path,
    cmm_backend: str = "lcms",
    transicc_path: str | Path = "transicc",
    xicclu_path: str = "xicclu",
) -> list[dict[str, object]]:
    if cmm_backend == "lcms":
        return convert_7clr_to_lab_lcms(rows, icc_path, transicc_path=transicc_path)
    if cmm_backend == "xicclu":
        return convert_7clr_to_lab_xicclu(rows, icc_path, xicclu_path=xicclu_path)
    raise ValueError(f"Unknown CMM backend: {cmm_backend}")


def convert_7clr_to_lab_lcms(
    rows: list[dict[str, object]],
    icc_path: str | Path,
    transicc_path: str | Path = "transicc",
    intent: int = 1,
) -> list[dict[str, object]]:
    profile = Path(icc_path)
    if not profile.exists():
        raise FileNotFoundError(f"ICC profile not found: {profile}")
    if not rows:
        return []

    command = [
        str(transicc_path),
        f"-i{profile}",
        "-o*Lab",
        f"-t{intent}",
        "-n",
    ]
    input_text = _build_7clr_input(rows)

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
            "transicc executable not found. Build LittleCMS transicc, ensure it is on PATH, "
            "or pass --transicc with the executable path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        stdout = exc.stdout.strip() if exc.stdout else ""
        detail = stderr or stdout
        raise CmmError(f"LittleCMS transicc conversion failed: {detail}") from exc

    labs = _parse_lab_output(result.stdout, expected_count=len(rows))
    converted: list[dict[str, object]] = []
    for row, lab in zip(rows, labs):
        merged = dict(row)
        merged.update(lab)
        converted.append(merged)
    return converted


def convert_7clr_to_lab_xicclu(
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
    input_text = _build_7clr_input(rows)

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
