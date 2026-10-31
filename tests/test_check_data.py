"""Test cho src/data/integrity.py bằng dữ liệu tổng hợp nhỏ (không cần dữ liệu thật)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.integrity import (
    DataError,
    compute_data_sha256,
    find_data_file,
    run_checks,
)

TRACK = "track1"
LABELED = ["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label"]
REFERENCE = ["instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "reverse_pair_id", "label"]


def make_rows(prefix: str) -> list[dict[str, str]]:
    """Hai cặp đảo được (FORWARD/BACKWARD, EQUIVALENCE/EQUIVALENCE) và một NEGATIVE_OTHER."""

    def row(iid, pid, t1, t2, rev, label):
        return {"instance_id": iid, "pair_id": pid, "text1_lang": "en", "text2_lang": "en",
                "text1": t1, "text2": t2, "reverse_pair_id": rev, "label": label}

    return [
        row(f"{prefix}1_a", f"{prefix}1", "a red car", "a car", f"{prefix}1_b", "FORWARD_ENTAILMENT"),
        row(f"{prefix}1_b", f"{prefix}1", "a car", "a red car", f"{prefix}1_a", "BACKWARD_ENTAILMENT"),
        row(f"{prefix}2_a", f"{prefix}2", "big house", "large house", f"{prefix}2_b", "EQUIVALENCE"),
        row(f"{prefix}2_b", f"{prefix}2", "large house", "big house", f"{prefix}2_a", "EQUIVALENCE"),
        row(f"{prefix}3_a", f"{prefix}3", "cold tea", "hot coffee", "", "NEGATIVE_OTHER"),
    ]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_split(raw_dir: Path, split: str, rows: list[dict[str, str]]) -> None:
    write_csv(raw_dir / f"dico_nli_{split}_{TRACK}_participant_labeled.csv", LABELED, rows)
    write_csv(raw_dir / f"dico_nli_{split}_{TRACK}_reference.csv", REFERENCE, rows)
    template = [{"instance_id": r["instance_id"], "label": ""} for r in rows]
    write_csv(raw_dir / f"dico_nli_{split}_{TRACK}_submission_template.csv", ["instance_id", "label"], template)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    write_split(tmp_path, "train", make_rows("tr"))
    write_split(tmp_path, "dev", make_rows("dv"))
    # Văn bản dev khác train để tránh cảnh báo trùng chéo split trong trường hợp hợp lệ
    return tmp_path


def run(raw_dir: Path, expected=None, splits=("train", "dev")):
    return run_checks(raw_dir, TRACK, expected or {"train": 5, "dev": 5}, splits)


def test_valid_data_has_no_errors(raw_dir: Path):
    report = run(raw_dir)
    assert report.errors == []


def test_wrong_row_count_is_error(raw_dir: Path):
    report = run(raw_dir, expected={"train": 6, "dev": 5})
    assert any("kỳ vọng 6" in e for e in report.errors)


def test_missing_file_is_error(raw_dir: Path):
    (raw_dir / f"dico_nli_dev_{TRACK}_reference.csv").unlink()
    report = run(raw_dir)
    assert any("thiếu file" in e for e in report.errors)


def test_unknown_label_is_error(tmp_path: Path):
    rows = make_rows("tr")
    rows[4]["label"] = "NEUTRAL"
    write_split(tmp_path, "train", rows)
    report = run(tmp_path, splits=("train",), expected={"train": 5})
    assert report.errors


def test_non_reciprocal_reverse_pair_is_error(tmp_path: Path):
    rows = make_rows("tr")
    rows[1]["reverse_pair_id"] = "tr2_a"  # trỏ sai
    write_split(tmp_path, "train", rows)
    report = run(tmp_path, splits=("train",), expected={"train": 5})
    assert any("scorer từ chối" in e for e in report.errors)


def test_incompatible_reverse_labels_is_error(tmp_path: Path):
    rows = make_rows("tr")
    rows[1]["label"] = "FORWARD_ENTAILMENT"  # đúng ra phải là BACKWARD
    write_split(tmp_path, "train", rows)
    report = run(tmp_path, splits=("train",), expected={"train": 5})
    assert any("scorer từ chối" in e for e in report.errors)


def test_reversible_label_without_reverse_id_is_error(tmp_path: Path):
    """Scorer bỏ qua âm thầm trường hợp này nên check_data phải bắt."""
    rows = make_rows("tr")
    rows[2]["reverse_pair_id"] = ""
    rows[3]["reverse_pair_id"] = ""
    write_split(tmp_path, "train", rows)
    report = run(tmp_path, splits=("train",), expected={"train": 5})
    assert any("thiếu reverse_pair_id" in e for e in report.errors)


def test_reverse_text_not_swapped_is_error(tmp_path: Path):
    rows = make_rows("tr")
    rows[1]["text1"], rows[1]["text2"] = "a car", "a blue car"
    write_split(tmp_path, "train", rows)
    report = run(tmp_path, splits=("train",), expected={"train": 5})
    assert any("không hoán đổi" in e for e in report.errors)


def test_labeled_and_reference_disagree_is_error(tmp_path: Path):
    rows = make_rows("tr")
    write_split(tmp_path, "train", rows)
    changed = [dict(r) for r in rows]
    changed[4]["text1"] = "something else"
    write_csv(tmp_path / f"dico_nli_train_{TRACK}_participant_labeled.csv", LABELED, changed)
    report = run(tmp_path, splits=("train",), expected={"train": 5})
    assert any("khác nhau giữa labeled và reference" in e for e in report.errors)


def test_pair_id_overlap_between_splits_is_error(tmp_path: Path):
    write_split(tmp_path, "train", make_rows("x"))
    write_split(tmp_path, "dev", make_rows("x"))
    report = run(tmp_path)
    assert any("pair_id xuất hiện ở cả hai split" in e for e in report.errors)


def test_repeated_text_is_warning_not_error(tmp_path: Path):
    rows = make_rows("tr")
    extra = dict(rows[4], instance_id="tr4_a", pair_id="tr4")
    rows.append(extra)
    write_split(tmp_path, "train", rows)
    report = run(tmp_path, splits=("train",), expected={"train": 6})
    assert report.errors == []
    assert any("lặp lại" in w for w in report.warnings)


def test_ambiguous_duplicate_file_raises(raw_dir: Path):
    name = f"dico_nli_dev_{TRACK}_reference.csv"
    (raw_dir / "copy").mkdir()
    (raw_dir / "copy" / name).write_text((raw_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(DataError):
        find_data_file(raw_dir, "dev", TRACK, "reference")


def test_data_hash_is_stable_and_sensitive_to_content(raw_dir: Path):
    first = compute_data_sha256(raw_dir, ("train", "dev"), TRACK)
    assert first == compute_data_sha256(raw_dir, ("train", "dev"), TRACK)
    path = raw_dir / f"dico_nli_dev_{TRACK}_reference.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert compute_data_sha256(raw_dir, ("train", "dev"), TRACK) != first
