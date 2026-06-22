from pathlib import Path

from . import INK_CHANNELS, TXT_TO_CHANNEL


def _find_line(lines: list[str], marker: str) -> int:
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return index
    raise ValueError(f"Missing marker: {marker}")


def _split_fields(line: str) -> list[str]:
    return line.strip().split()


def parse_7clr_txt(path: str | Path) -> list[dict[str, object]]:
    txt_path = Path(path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Txt chart not found: {txt_path}")

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    format_start = _find_line(lines, "BEGIN_DATA_FORMAT")
    format_end = _find_line(lines, "END_DATA_FORMAT")
    data_start = _find_line(lines, "BEGIN_DATA")
    data_end = _find_line(lines, "END_DATA")

    if format_end <= format_start + 1:
        raise ValueError("No data format row found")
    if data_end <= data_start + 1:
        raise ValueError("No data rows found")

    fields = _split_fields(lines[format_start + 1])
    required = ["SampleID", "SAMPLE_NAME", *TXT_TO_CHANNEL.keys()]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines[data_start + 1 : data_end], start=data_start + 2):
        if not line.strip():
            continue

        values = _split_fields(line)
        if len(values) != len(fields):
            raise ValueError(
                f"Malformed data row at line {line_number}: expected {len(fields)} fields, got {len(values)}"
            )

        raw = dict(zip(fields, values))
        row: dict[str, object] = {
            "SampleID": raw["SampleID"],
            "SAMPLE_NAME": raw["SAMPLE_NAME"],
        }

        for txt_field, channel in TXT_TO_CHANNEL.items():
            value = float(raw[txt_field])
            if value < 0 or value > 100:
                raise ValueError(
                    f"Channel {txt_field} for sample {row['SAMPLE_NAME']} is outside 0-100: {value}"
                )
            row[channel] = value

        rows.append(row)

    return rows
