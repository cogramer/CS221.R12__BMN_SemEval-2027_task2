"""Kiểm tra toàn vẹn dữ liệu DiCo-NLI (final_data) và băm dữ liệu.

Dùng chung cho scripts/check_data.py, tests/ và (sau này) run_seed.sh
để ghi `data_sha256`. Không phụ thuộc torch/transformers.

Nguyên tắc: lỗi (error) làm script thất bại; cảnh báo (warning) chỉ báo cáo,
vì việc xử lý dữ liệu trùng là quyết định của cả nhóm.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"

KIND_LABELED = "participant_labeled"
KIND_REFERENCE = "reference"
KIND_TEMPLATE = "submission_template"

LABELED_COLUMNS = ("instance_id", "pair_id", "text1_lang", "text2_lang", "text1", "text2", "label")
REFERENCE_COLUMNS = ("instance_id", "pair_id", "reverse_pair_id", "label")
TEMPLATE_COLUMNS = ("instance_id", "label")


class DataError(Exception):
    """Lỗi nghiêm trọng khiến không thể tiếp tục kiểm tra (thiếu file, file trùng tên)."""


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def ensure_scorer_importable() -> None:
    """Thêm third_party/ vào sys.path để import được `evaluation_functions`."""
    path = str(THIRD_PARTY)
    if path not in sys.path:
        sys.path.insert(0, path)


def data_filename(split: str, track: str, kind: str) -> str:
    return f"dico_nli_{split}_{track}_{kind}.csv"


def find_data_file(raw_dir: Path, split: str, track: str, kind: str) -> Path | None:
    """Tìm file theo tên, kể cả khi nằm trong thư mục con. Trả None nếu không có."""
    name = data_filename(split, track, kind)
    matches = sorted(p for p in Path(raw_dir).rglob(name) if p.is_file())
    if len(matches) > 1:
        raise DataError(f"Có {len(matches)} file trùng tên {name} dưới {raw_dir}: {[str(m) for m in matches]}")
    return matches[0] if matches else None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Đọc CSV bằng thư viện csv (không dùng pandas để tránh biến "NA"/"None" thành NaN)."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def compute_data_sha256(raw_dir: Path, splits: tuple[str, ...], track: str) -> str:
    """Băm nội dung file labeled + reference của các split (theo tên file, không theo đường dẫn)."""
    digest = hashlib.sha256()
    names: list[tuple[str, Path]] = []
    for split in splits:
        for kind in (KIND_LABELED, KIND_REFERENCE):
            path = find_data_file(raw_dir, split, track, kind)
            if path is None:
                raise DataError(f"Thiếu file {data_filename(split, track, kind)} dưới {raw_dir}.")
            names.append((path.name, path))
    for name, path in sorted(names):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _check_columns(report: Report, tag: str, fieldnames: list[str], required: tuple[str, ...], exact: bool) -> bool:
    missing = [c for c in required if c not in fieldnames]
    if missing:
        report.errors.append(f"[{tag}] thiếu cột: {missing}")
        return False
    if exact and set(fieldnames) != set(required):
        report.errors.append(f"[{tag}] cột phải đúng {list(required)}, thực tế {fieldnames}")
        return False
    if len(set(fieldnames)) != len(fieldnames):
        report.errors.append(f"[{tag}] có cột trùng tên: {fieldnames}")
        return False
    return True


def check_split(raw_dir: Path, split: str, track: str, expected_rows: int, report: Report):
    """Kiểm tra một split. Trả về danh sách dòng labeled (hoặc None nếu không đọc được)."""
    ensure_scorer_importable()
    from evaluation_functions.errors import ScoringError
    from evaluation_functions.io import read_reference_file
    from evaluation_functions.labels import ALL_LABELS, is_reversible_label, reverse_label
    from evaluation_functions.validation import build_reversible_pairs, validate_reference_records

    tag = split
    paths = {kind: find_data_file(raw_dir, split, track, kind) for kind in (KIND_LABELED, KIND_REFERENCE, KIND_TEMPLATE)}
    absent = [data_filename(split, track, k) for k, p in paths.items() if p is None]
    if absent:
        report.errors.append(f"[{tag}] thiếu file: {absent}")
        return None

    lab_cols, labeled = read_csv_rows(paths[KIND_LABELED])
    ref_cols, reference = read_csv_rows(paths[KIND_REFERENCE])
    tpl_cols, template = read_csv_rows(paths[KIND_TEMPLATE])

    ok = _check_columns(report, f"{tag}/labeled", lab_cols, LABELED_COLUMNS, exact=False)
    ok &= _check_columns(report, f"{tag}/reference", ref_cols, REFERENCE_COLUMNS, exact=False)
    ok &= _check_columns(report, f"{tag}/template", tpl_cols, TEMPLATE_COLUMNS, exact=True)
    if not ok:
        return None

    # 1. Số dòng
    for kind, rows in ((KIND_LABELED, labeled), (KIND_REFERENCE, reference), (KIND_TEMPLATE, template)):
        if len(rows) != expected_rows:
            report.errors.append(f"[{tag}/{kind}] có {len(rows)} dòng, kỳ vọng {expected_rows}")

    # 2. Nhãn hợp lệ, id duy nhất, văn bản không rỗng (file labeled)
    seen: set[str] = set()
    for i, row in enumerate(labeled, start=2):
        if row["label"] not in ALL_LABELS:
            report.errors.append(f"[{tag}/labeled] dòng {i}: nhãn lạ {row['label']!r}")
        if row["instance_id"] in seen:
            report.errors.append(f"[{tag}/labeled] dòng {i}: trùng instance_id {row['instance_id']!r}")
        seen.add(row["instance_id"])
        if not row["text1"].strip() or not row["text2"].strip():
            report.errors.append(f"[{tag}/labeled] dòng {i}: text1 hoặc text2 rỗng")

    # 3. Rev(Rev(y)) == y trên bảng nhãn của scorer
    for label in ALL_LABELS:
        if is_reversible_label(label) and reverse_label(reverse_label(label)) != label:
            report.errors.append(f"[{tag}] bảng nhãn đảo của scorer không phải phép đối hợp tại {label}")

    # 4. Scorer tự kiểm tra reference (đối xứng, cùng pair_id, nhãn đảo khớp, không rỗng...)
    gold = None
    try:
        gold = validate_reference_records(read_reference_file(paths[KIND_REFERENCE]))
    except ScoringError as exc:
        report.errors.append(f"[{tag}/reference] scorer từ chối: {exc}")

    # 5. Nhãn đảo được thì BẮT BUỘC có reverse_pair_id (scorer bỏ qua âm thầm nếu thiếu)
    #    và nhãn không đảo được thì không được có.
    ref_by_id = {row["instance_id"]: row for row in reference}
    for row in reference:
        rev_id = row["reverse_pair_id"].strip()
        reversible = is_reversible_label(row["label"]) if row["label"] in ALL_LABELS else None
        if reversible is True and not rev_id:
            report.errors.append(f"[{tag}/reference] {row['instance_id']}: nhãn {row['label']} nhưng thiếu reverse_pair_id")
        if reversible is False and rev_id:
            report.errors.append(f"[{tag}/reference] {row['instance_id']}: NEGATIVE_OTHER nhưng có reverse_pair_id")

    # 6. labeled khớp reference (id, pair_id, nhãn, văn bản)
    if set(ref_by_id) != seen:
        report.errors.append(
            f"[{tag}] tập instance_id của labeled và reference khác nhau "
            f"(chỉ ở labeled: {len(seen - set(ref_by_id))}, chỉ ở reference: {len(set(ref_by_id) - seen)})"
        )
    for row in labeled:
        ref = ref_by_id.get(row["instance_id"])
        if ref is None:
            continue
        for column in ("pair_id", "label"):
            if row[column] != ref[column]:
                report.errors.append(f"[{tag}] {row['instance_id']}: {column} khác nhau giữa labeled và reference")
        for column in ("text1", "text2"):
            if column in ref and row[column] != ref[column]:
                report.errors.append(f"[{tag}] {row['instance_id']}: {column} khác nhau giữa labeled và reference")

    # 7. Template có đúng tập instance_id
    tpl_ids = [row["instance_id"] for row in template]
    if len(set(tpl_ids)) != len(tpl_ids) or set(tpl_ids) != seen:
        report.errors.append(f"[{tag}/template] tập instance_id không khớp file labeled")

    # 8. Cặp đảo phải hoán đổi đúng văn bản
    labeled_by_id = {row["instance_id"]: row for row in labeled}
    for row in labeled:
        ref = ref_by_id.get(row["instance_id"])
        rev_id = ref["reverse_pair_id"].strip() if ref else ""
        other = labeled_by_id.get(rev_id) if rev_id else None
        if other is not None and (row["text1"] != other["text2"] or row["text2"] != other["text1"]):
            report.errors.append(f"[{tag}] {row['instance_id']} và {rev_id}: văn bản không hoán đổi đúng")

    # 9. Thông tin và cảnh báo
    distribution: dict[str, int] = defaultdict(int)
    for row in labeled:
        distribution[row["label"]] += 1
    report.info.append(f"[{tag}] {len(labeled)} dòng; nhãn: {dict(sorted(distribution.items()))}")
    if gold is not None:
        try:
            report.info.append(f"[{tag}] {len(build_reversible_pairs(gold))} cặp đảo được (dùng cho SoftCons/HardCons)")
        except ScoringError as exc:
            report.errors.append(f"[{tag}] không dựng được cặp đảo: {exc}")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in labeled:
        groups[(row["text1"], row["text2"])].append(row)
    repeated = {key: rows for key, rows in groups.items() if len(rows) > 1}
    if repeated:
        extra_rows = sum(len(rows) - 1 for rows in repeated.values())
        conflicting = sum(1 for rows in repeated.values() if len({r["label"] for r in rows}) > 1)
        report.warnings.append(
            f"[{tag}] {len(repeated)} cặp văn bản (text1, text2) lặp lại ({extra_rows} dòng dư, "
            f"khác pair_id); {conflicting} cặp có nhãn mâu thuẫn"
        )
    return labeled


def check_cross_split(labeled_by_split: dict[str, list[dict[str, str]]], report: Report) -> None:
    """So sánh giữa các split: trùng instance_id/pair_id là lỗi, trùng văn bản là cảnh báo."""
    splits = sorted(labeled_by_split)
    for i, first in enumerate(splits):
        for second in splits[i + 1:]:
            a, b = labeled_by_split[first], labeled_by_split[second]
            shared_ids = {r["instance_id"] for r in a} & {r["instance_id"] for r in b}
            shared_pairs = {r["pair_id"] for r in a} & {r["pair_id"] for r in b}
            shared_text = {(r["text1"], r["text2"]) for r in a} & {(r["text1"], r["text2"]) for r in b}
            if shared_ids:
                report.errors.append(f"[{first} x {second}] {len(shared_ids)} instance_id xuất hiện ở cả hai split")
            if shared_pairs:
                report.errors.append(f"[{first} x {second}] {len(shared_pairs)} pair_id xuất hiện ở cả hai split")
            if shared_text:
                report.warnings.append(f"[{first} x {second}] {len(shared_text)} cặp văn bản (text1, text2) xuất hiện ở cả hai split")


def run_checks(raw_dir: Path, track: str, expected_rows: dict[str, int], splits: tuple[str, ...]) -> Report:
    """Chạy toàn bộ kiểm tra cho các split được chọn."""
    report = Report()
    labeled_by_split: dict[str, list[dict[str, str]]] = {}
    for split in splits:
        if split not in expected_rows:
            report.errors.append(f"[{split}] chưa khai báo expected_rows trong cấu hình")
            continue
        try:
            labeled = check_split(Path(raw_dir), split, track, expected_rows[split], report)
        except DataError as exc:
            report.errors.append(f"[{split}] {exc}")
            continue
        if labeled is not None:
            labeled_by_split[split] = labeled
    if len(labeled_by_split) > 1:
        check_cross_split(labeled_by_split, report)
    return report
