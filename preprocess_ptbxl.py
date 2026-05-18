import os
import ast
import argparse
import numpy as np
import pandas as pd
import wfdb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True,
                        help="PTB-XL root directory")
    parser.add_argument("--sampling_rate", type=int, default=100, choices=[100, 500],
                        help="Use 100 or 500 Hz records")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory")
    return parser.parse_args()


def load_ptbxl_metadata(src_dir: str) -> pd.DataFrame:
    meta_path = os.path.join(src_dir, "ptbxl_database.csv")
    meta = pd.read_csv(meta_path)

    # scp_codes는 문자열 dict라 파싱해두면 나중에 편함
    if "scp_codes" in meta.columns:
        meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

    return meta


def choose_filename_column(sampling_rate: int) -> str:
    if sampling_rate == 100:
        return "filename_lr"
    if sampling_rate == 500:
        return "filename_hr"
    raise ValueError(f"Unsupported sampling_rate: {sampling_rate}")


def load_signals(meta: pd.DataFrame, src_dir: str, file_col: str) -> np.ndarray:
    signals = []

    for i, rel_path in enumerate(meta[file_col].tolist()):
        record_path = os.path.join(src_dir, rel_path)
        sig, _ = wfdb.rdsamp(record_path)
        signals.append(sig.astype(np.float32))

        if (i + 1) % 1000 == 0:
            print(f"[{i + 1}/{len(meta)}] loaded")

    # PTB-XL은 고정 길이이므로 stack 가능
    x = np.stack(signals, axis=0)  # (N, T, C)
    return x


def build_meta(meta: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    # 내부 인덱스
    out["sample_id"] = np.arange(len(meta), dtype=np.int64)

    # 원본 추적용
    for col in ["ecg_id", "patient_id", "age", "sex", "height", "weight",
                "nurse", "site", "device", "recording_date", "report",
                "strat_fold"]:
        if col in meta.columns:
            out[col] = meta[col]

    # fold 기반 subset 표기
    if "strat_fold" in meta.columns:
        def fold_to_subset(fold: int) -> str:
            if 1 <= fold <= 8:
                return "train"
            if fold == 9:
                return "val"
            if fold == 10:
                return "test"
            return "unknown"

        out["subset"] = meta["strat_fold"].apply(fold_to_subset)

    return out


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    meta = load_ptbxl_metadata(args.src_dir)
    file_col = choose_filename_column(args.sampling_rate)

    x = load_signals(meta, args.src_dir, file_col)
    out_meta = build_meta(meta)

    data_path = os.path.join(args.save_dir, "ptbxl_data.npy")
    meta_path = os.path.join(args.save_dir, "ptbxl_meta.csv")

    np.save(data_path, x)
    out_meta.to_csv(meta_path, index=False)

    print("Saved:")
    print(f"  data: {data_path}  shape={x.shape} dtype={x.dtype}")
    print(f"  meta: {meta_path}")


if __name__ == "__main__":
    main()