#!/usr/bin/env python3
"""Kiểm tra dữ liệu final_data trước khi huấn luyện.

Ví dụ:
    python scripts/check_data.py
    python scripts/check_data.py --splits dev
    python scripts/check_data.py --print-hash
    python scripts/check_data.py --strict      # cảnh báo cũng làm script thất bại

Mã thoát: 0 nếu không có lỗi, 1 nếu có lỗi (hoặc có cảnh báo khi dùng --strict).
Số dòng kỳ vọng và đường dẫn lấy từ configs/base.yaml (mục data).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from src.data.integrity import DataError, compute_data_sha256, run_checks  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--raw-dir", default=None, help="Mặc định lấy data.raw_dir trong cấu hình")
    parser.add_argument("--splits", nargs="+", default=None, help="Mặc định: mọi split trong data.expected_rows")
    parser.add_argument("--strict", action="store_true", help="Coi cảnh báo là lỗi")
    parser.add_argument("--print-hash", action="store_true", help="In data_sha256 nếu kiểm tra không có lỗi")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_cfg = config["data"]

    raw_dir = Path(args.raw_dir) if args.raw_dir else REPO_ROOT / data_cfg["raw_dir"]
    expected_rows = {str(k): int(v) for k, v in data_cfg["expected_rows"].items()}
    splits = tuple(args.splits) if args.splits else tuple(expected_rows)
    track = data_cfg["track"]

    report = run_checks(raw_dir, track, expected_rows, splits)

    for line in report.info:
        print(f"INFO     {line}")
    for line in report.warnings:
        print(f"WARNING  {line}")
    for line in report.errors:
        print(f"ERROR    {line}")

    failed = bool(report.errors) or (args.strict and bool(report.warnings))
    print()
    print(f"Kết quả: {'THẤT BẠI' if failed else 'ĐẠT'} "
          f"({len(report.errors)} lỗi, {len(report.warnings)} cảnh báo)")

    if args.print_hash and not report.errors:
        try:
            print(f"data_sha256: {compute_data_sha256(raw_dir, splits, track)}")
        except DataError as exc:
            print(f"ERROR    không tính được data_sha256: {exc}")
            return 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
