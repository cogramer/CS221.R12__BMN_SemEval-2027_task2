# CS221.R12__BMN_SemEval-2027_task2

SemEval-2027 Task 2. Backbone: `microsoft/deberta-v3-base`.
Phương pháp: đối xứng hóa lúc suy luận (P_sym), ép nhất quán lúc huấn luyện,
augmentation đảo cặp.

## Cài đặt

    git clone <url-repo> && cd CS221.R12__BMN_SemEval-2027_task2
    pip install -r requirements.txt

## Dữ liệu

Đặt `final_data` gốc vào `data/raw/` (không sửa, không commit).
Kiểm tra: `python scripts/check_data.py` (train 3.042, dev 660, 4 nhãn).
Sinh dữ liệu đảo cặp vào `data/processed/` bằng script trong `src/data/`.

## Chạy một cấu hình

    bash scripts/run_seed.sh configs/E5.yaml 42     # một seed
    bash scripts/run_seed.sh configs/E5.yaml 42 --verify   # kiểm tra tái lập
    bash scripts/run_sweep.sh                       # mọi cấu hình x seed (42, 43, 44)
    python scripts/aggregate.py                     # bảng tổng hợp

Kết quả: `results/<cấu_hình>/seed_<n>/` (`metrics.json`, `predictions.csv`,
`probs.npy`, `config_used.yaml`, `log.txt`).
Bảng trong paper chỉ lấy từ `aggregate.py`. Notebook không phải nguồn kết quả chính thức.

## Cấu hình thí nghiệm

| Cấu hình | Nội dung |
|---|---|
| E0 | DeBERTa thường, mốc gốc |
| E1 | Chỉ augmentation đảo cặp |
| E2 | Chỉ đối xứng hóa lúc suy luận |
| E3 | Chỉ ép nhất quán |
| E4 | Đối xứng hóa + augmentation |
| E5 / E6 / E7 | Cấu hình chính, λ = 0.5 / 1 / 0.1 |
| E8 | Đối chứng `cross-encoder/nli-deberta-v3-base` |

## Quy tắc chính

- Chỉ dùng seed 42, 43, 44. Không sửa cấu hình đã có kết quả (tạo biến thể `E5b`).
- Không dùng test để chọn siêu tham số. Test bị khóa tới khi có file `TEST_UNLOCKED`.
- Thứ tự nhãn chỉ định nghĩa trong `src/model/labels.py`.
- Merge vào `main` qua PR, cần test qua và một người review (A → B → C → A).

## Mô hình và dữ liệu bên ngoài (để khai báo trong paper)

- `microsoft/deberta-v3-base` (Hugging Face)
- `cross-encoder/nli-deberta-v3-base` (E8; được huấn luyện trên SNLI/MNLI)

## Phân công

- A: dữ liệu và nền tảng repo (`src/data/`)
- B: mô hình và huấn luyện (`src/model/`, `src/train/`)
- C: đánh giá, baseline, phân tích lỗi (`src/eval/`, `src/baselines/`)
