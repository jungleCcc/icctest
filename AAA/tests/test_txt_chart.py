from pathlib import Path

import pytest

from icc_validation.txt_chart import INK_CHANNELS, parse_7clr_txt


def write_chart(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "chart.txt"
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_7clr_txt_maps_fields_to_named_channels(tmp_path):
    chart = write_chart(
        tmp_path,
        "\n".join(
            [
                'ORIGINATOR\t"X GAMUT"',
                "NUMBER_OF_FIELDS 9",
                "BEGIN_DATA_FORMAT",
                "SampleID\tSAMPLE_NAME\t7CLR_1\t7CLR_2\t7CLR_3\t7CLR_4\t7CLR_5\t7CLR_6\t7CLR_7",
                "END_DATA_FORMAT",
                "NUMBER_OF_SETS\t2",
                "BEGIN_DATA",
                "1\tA1\t20.00\t0.00\t0.00\t100.00\t0.00\t0.00\t0.00",
                "2\tA2\t0.00\t70.00\t0.00\t100.00\t0.00\t0.00\t0.00",
                "END_DATA",
            ]
        ),
    )

    rows = parse_7clr_txt(chart)

    assert len(rows) == 2
    assert rows[0]["SampleID"] == "1"
    assert rows[0]["SAMPLE_NAME"] == "A1"
    assert rows[0]["Cyan"] == 20.0
    assert rows[0]["Magenta"] == 0.0
    assert rows[0]["Yellow"] == 0.0
    assert rows[0]["Black"] == 100.0
    assert rows[0]["Orange"] == 0.0
    assert rows[0]["Green"] == 0.0
    assert rows[0]["Violet"] == 0.0
    assert tuple(INK_CHANNELS) == (
        "Cyan",
        "Magenta",
        "Yellow",
        "Black",
        "Orange",
        "Green",
        "Violet",
    )


def test_parse_7clr_txt_rejects_missing_channel(tmp_path):
    chart = write_chart(
        tmp_path,
        "\n".join(
            [
                "BEGIN_DATA_FORMAT",
                "SampleID\tSAMPLE_NAME\t7CLR_1\t7CLR_2",
                "END_DATA_FORMAT",
                "BEGIN_DATA",
                "1\tA1\t20.00\t0.00",
                "END_DATA",
            ]
        ),
    )

    with pytest.raises(ValueError, match="Missing required fields"):
        parse_7clr_txt(chart)


def test_parse_7clr_txt_rejects_out_of_range_channel(tmp_path):
    chart = write_chart(
        tmp_path,
        "\n".join(
            [
                "BEGIN_DATA_FORMAT",
                "SampleID\tSAMPLE_NAME\t7CLR_1\t7CLR_2\t7CLR_3\t7CLR_4\t7CLR_5\t7CLR_6\t7CLR_7",
                "END_DATA_FORMAT",
                "BEGIN_DATA",
                "1\tA1\t101.00\t0.00\t0.00\t100.00\t0.00\t0.00\t0.00",
                "END_DATA",
            ]
        ),
    )

    with pytest.raises(ValueError, match="outside 0-100"):
        parse_7clr_txt(chart)
