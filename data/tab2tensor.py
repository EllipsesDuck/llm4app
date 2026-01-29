import gc
import json
import os
import glob
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


ROOT = r"E:/NUS/data/perdata/train_text_all_samples"

IN_TAB_DIR = os.path.join(ROOT, "tabular")
OUT_TAB_DIR = os.path.join(ROOT, "tabular_pruned")

IN_TAB_BUCKET_DIR = os.path.join(ROOT, "tabular_clean_v1")
OUT_DIR = os.path.join(ROOT, "tab_tensor_v1")

NUM_BUCKETS = 256
CHUNKSIZE = 300_000
CHUNKSIZE_DEFAULT = 1_000_000

MISSING_TOKEN = "__MISSING__"

TABLES: List[str] = [
    "demographics",
    # "admissions",
    "icustays",
    "transfers",
    "prescriptions",
    # "procedures_icd",
]

TIME_COL: Dict[str, Optional[str]] = {
    "demographics": None,
    # "admissions": "admittime",
    "icustays": "intime",
    "transfers": "intime",
    "prescriptions": "starttime",
    # "procedures_icd": "chartdate",
}

CAT_COLS: List[str] = [
    "gender",
    "anchor_year_group",
    # "admission_type",
    # "admission_location",
    # "insurance",
    # "language",
    # "marital_status",
    "first_careunit",
    "last_transfer_eventtype",
    "last_transfer_careunit",
    "top_drug_type",
    "top_route",
    # "top_drug",
    # "top_icd_code",
]

NUM_COLS: List[str] = [
    "anchor_age",
    # "hadm_count",
    "icustay_count",
    "transfer_count",
    "presc_count",
    "presc_unique_drug",
    # "proc_count",
    # "proc_unique_icd",
    # "icu_los_hours",
]


def _clean_cat(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    s = s.fillna(MISSING_TOKEN)
    s = s.replace("", MISSING_TOKEN)
    return s


def _ensure_datetime(df: pd.DataFrame, col: Optional[str]) -> pd.DataFrame:
    if col and col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def find_table_file(tab_dir: str, name: str) -> str:
    candidates: List[str] = []
    for ext in ["csv", "csv.gz", "tsv", "tsv.gz"]:
        candidates += glob.glob(os.path.join(tab_dir, f"{name}.{ext}"))
    if not candidates:
        candidates = glob.glob(os.path.join(tab_dir, f"{name}.*"))
    if not candidates:
        raise FileNotFoundError(f"Cannot find file for table={name} under {tab_dir}")
    candidates = sorted(candidates, key=lambda x: (len(os.path.basename(x)), x))
    return candidates[0]


def safe_cast_chunk(df: pd.DataFrame, dtype_map: Optional[dict], parse_dates: Optional[List[str]]):
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    if dtype_map:
        for c, dt in dtype_map.items():
            if c not in df.columns:
                continue
            if dt == "category":
                df[c] = df[c].astype("string")
            else:
                if "int" in str(dt) or "float" in str(dt):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                try:
                    df[c] = df[c].astype(dt)
                except Exception:
                    pass
    return df


def prune_tabular() -> None:
    os.makedirs(OUT_TAB_DIR, exist_ok=True)

    try:
        from config import TAB_CFG
    except Exception as e:
        raise RuntimeError("Failed to import TAB_CFG from config.py") from e

    for name, cfg in TAB_CFG.items():
        in_fp = find_table_file(IN_TAB_DIR, name)
        out_fp = os.path.join(OUT_TAB_DIR, f"{name}.csv")

        usecols = cfg.get("usecols", None)
        dtype_map = cfg.get("dtype", None)
        parse_dates = cfg.get("parse_dates", None)

        fsize_mb = os.path.getsize(in_fp) / (1024 * 1024)
        use_chunks = fsize_mb >= 200

        print(
            f"[RUN] {name}: {os.path.basename(in_fp)} ({fsize_mb:.1f} MB) -> "
            f"{os.path.basename(out_fp)} | chunks={use_chunks}"
        )

        if os.path.exists(out_fp):
            os.remove(out_fp)

        if use_chunks:
            reader = pd.read_csv(
                in_fp,
                usecols=usecols,
                low_memory=False,
                chunksize=CHUNKSIZE_DEFAULT,
            )
            first = True
            for chunk in reader:
                chunk = safe_cast_chunk(chunk, dtype_map, parse_dates)
                chunk.to_csv(out_fp, mode="a", index=False, header=first)
                first = False
        else:
            df = pd.read_csv(in_fp, usecols=usecols, low_memory=False)
            df = safe_cast_chunk(df, dtype_map, parse_dates)
            df.to_csv(out_fp, index=False)

        print(f"[OK ] saved: {out_fp}")

    print(f"[DONE] pruned tabular saved to: {OUT_TAB_DIR}")


def bucket_file(table: str, bucket: int) -> str:
    return os.path.join(IN_TAB_BUCKET_DIR, table, f"bucket_{bucket:03d}.csv")


def read_bucket_csv(path: str, chunksize: int) -> pd.io.parsers.TextFileReader:
    return pd.read_csv(
        path,
        low_memory=False,
        chunksize=chunksize,
        on_bad_lines="skip",
    )


def pick_latest_row(df: pd.DataFrame, time_col: Optional[str]) -> pd.DataFrame:
    if time_col is None or time_col not in df.columns:
        return df.sort_values(["sample_id"]).drop_duplicates(["sample_id"], keep="first")

    df = _ensure_datetime(df, time_col)
    df["_t"] = df[time_col].view("int64")
    df["_t"] = df["_t"].fillna(-1)
    df = df.sort_values(["sample_id", "_t"], ascending=[True, True])
    out = df.drop_duplicates(["sample_id"], keep="last").drop(columns=["_t"], errors="ignore")
    return out


def agg_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sample_id"]).drop_duplicates(["sample_id"], keep="first")
    cols = ["sample_id", "subject_id", "gender", "anchor_age", "anchor_year_group"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def agg_admissions(df: pd.DataFrame) -> pd.DataFrame:
    df = pick_latest_row(df, "admittime")
    keep = [
        "sample_id",
        "hadm_id",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def agg_icustays(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime(df, "intime")
    df = _ensure_datetime(df, "outtime")

    df_latest = pick_latest_row(df, "intime")
    base_cols = ["sample_id", "first_careunit", "intime", "outtime"]
    base_cols = [c for c in base_cols if c in df_latest.columns]
    out = df_latest[base_cols].copy()

    if "intime" in out.columns and "outtime" in out.columns:
        out["icu_los_hours"] = (out["outtime"] - out["intime"]).dt.total_seconds() / 3600.0
        out["icu_los_hours"] = out["icu_los_hours"].replace([np.inf, -np.inf], np.nan)
    else:
        out["icu_los_hours"] = np.nan

    cnt = df.groupby("sample_id", sort=False).size().rename("icustay_count").reset_index()
    out = out.merge(cnt, on="sample_id", how="left")
    out = out.drop(columns=["intime", "outtime"], errors="ignore")
    return out


def agg_transfers(df: pd.DataFrame) -> pd.DataFrame:
    cnt = df.groupby("sample_id", sort=False).size().rename("transfer_count").reset_index()

    df_latest = pick_latest_row(df, "intime")
    cols = ["sample_id", "eventtype", "careunit"]
    cols = [c for c in cols if c in df_latest.columns]
    out = df_latest[cols].copy()

    out = out.rename(
        columns={
            "eventtype": "last_transfer_eventtype",
            "careunit": "last_transfer_careunit",
        }
    )
    out = out.merge(cnt, on="sample_id", how="left")
    return out


def _top1_str(series: pd.Series) -> str:
    series = series.dropna().astype(str)
    if len(series) == 0:
        return MISSING_TOKEN
    vc = series.value_counts()
    return str(vc.index[0]) if len(vc) else MISSING_TOKEN


def agg_prescriptions(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sample_id", sort=False)
    out = g.size().rename("presc_count").reset_index()

    if "drug" in df.columns:
        out2 = g["drug"].nunique(dropna=True).rename("presc_unique_drug").reset_index()
        out = out.merge(out2, on="sample_id", how="left")
    else:
        out["presc_unique_drug"] = 0

    for src, dst in [("drug_type", "top_drug_type"), ("route", "top_route"), ("drug", "top_drug")]:
        if src in df.columns:
            top = g[src].apply(_top1_str).rename(dst).reset_index()
            out = out.merge(top, on="sample_id", how="left")
        else:
            out[dst] = MISSING_TOKEN

    return out


def agg_procedures(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("sample_id", sort=False)
    out = g.size().rename("proc_count").reset_index()

    if "icd_code" in df.columns:
        out2 = g["icd_code"].nunique(dropna=True).rename("proc_unique_icd").reset_index()
        out = out.merge(out2, on="sample_id", how="left")
        top = g["icd_code"].apply(_top1_str).rename("top_icd_code").reset_index()
        out = out.merge(top, on="sample_id", how="left")
    else:
        out["proc_unique_icd"] = 0
        out["top_icd_code"] = MISSING_TOKEN

    return out


AGG_FN = {
    "demographics": agg_demographics,
    # "admissions": agg_admissions,
    "icustays": agg_icustays,
    "transfers": agg_transfers,
    "prescriptions": agg_prescriptions,
    # "procedures_icd": agg_procedures,
}


def _bucket_has_any_table(bucket: int) -> bool:
    for t in TABLES:
        if os.path.exists(bucket_file(t, bucket)):
            return True
    return False


def _read_table_bucket(table: str, bucket: int) -> Optional[pd.DataFrame]:
    fp = bucket_file(table, bucket)
    if not os.path.exists(fp):
        return None

    parts: List[pd.DataFrame] = []
    for chunk in read_bucket_csv(fp, chunksize=CHUNKSIZE):
        if "sample_id" not in chunk.columns:
            continue
        chunk["sample_id"] = _clean_cat(chunk["sample_id"])
        parts.append(chunk)

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    del parts
    return df


def _merge_bucket_tables(bucket: int) -> Optional[pd.DataFrame]:
    bucket_df: Optional[pd.DataFrame] = None

    for t in TABLES:
        df = _read_table_bucket(t, bucket)
        if df is None:
            continue

        df_agg = AGG_FN[t](df)

        if bucket_df is None:
            bucket_df = df_agg
        else:
            bucket_df = bucket_df.merge(df_agg, on="sample_id", how="outer")

        del df, df_agg
        gc.collect()

    return bucket_df


def build_vocab_all_buckets() -> Dict[str, Dict[str, int]]:
    vocabs: Dict[str, Dict[str, int]] = {c: {MISSING_TOKEN: 0} for c in CAT_COLS}

    for b in range(NUM_BUCKETS):
        if not _bucket_has_any_table(b):
            continue

        print(f"[VOCAB] scanning bucket {b:03d}")

        bucket_df = _merge_bucket_tables(b)
        if bucket_df is None or len(bucket_df) == 0:
            continue

        for c in CAT_COLS:
            if c not in bucket_df.columns:
                continue
            col = _clean_cat(bucket_df[c])
            for tok in col.dropna().unique().tolist():
                tok = str(tok)
                if tok not in vocabs[c]:
                    vocabs[c][tok] = len(vocabs[c])

        del bucket_df
        gc.collect()

    print("[VOCAB] done.")
    return vocabs


def make_sample_level_bucket(bucket: int) -> Optional[pd.DataFrame]:
    bucket_df = _merge_bucket_tables(bucket)
    if bucket_df is None or len(bucket_df) == 0:
        return None

    # if "hadm_id" in bucket_df.columns:
    #     bucket_df["hadm_count"] = bucket_df["hadm_id"].notna().astype(np.float32)
    # else:
    #     bucket_df["hadm_count"] = 0.0

    for c in NUM_COLS:
        if c not in bucket_df.columns:
            bucket_df[c] = 0.0

    for c in CAT_COLS:
        if c not in bucket_df.columns:
            bucket_df[c] = MISSING_TOKEN
        bucket_df[c] = _clean_cat(bucket_df[c])

    if "anchor_age" in bucket_df.columns:
        anchor_age = pd.to_numeric(bucket_df["anchor_age"], errors="coerce").fillna(0.0)
    else:
        anchor_age = pd.Series(np.zeros(len(bucket_df), dtype=np.float32))
    bucket_df["anchor_age"] = anchor_age.clip(lower=0, upper=120)

    for c in NUM_COLS:
        bucket_df[c] = pd.to_numeric(bucket_df[c], errors="coerce").fillna(0.0).astype(np.float32)

    bucket_df = bucket_df.sort_values("sample_id").reset_index(drop=True)
    return bucket_df


def encode_and_dump(bucket: int, vocabs: Dict[str, Dict[str, int]]) -> None:
    df = make_sample_level_bucket(bucket)
    if df is None:
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    out_b_dir = os.path.join(OUT_DIR, f"bucket_{bucket:03d}")
    os.makedirs(out_b_dir, exist_ok=True)

    sample_ids = df["sample_id"].astype(str).values
    np.save(os.path.join(out_b_dir, "sample_id.npy"), sample_ids)

    X_num = df[NUM_COLS].to_numpy(dtype=np.float32, copy=True)
    np.save(os.path.join(out_b_dir, "numeric.npy"), X_num)

    for i, c in enumerate(CAT_COLS):
        vocab = vocabs[c]
        idx = df[c].map(lambda x: vocab.get(str(x), 0)).astype(np.int64).values
        np.save(os.path.join(out_b_dir, f"cat_{i:02d}_{c}.npy"), idx)

    schema = {
        "bucket": bucket,
        "num_rows": int(len(df)),
        "num_cols": NUM_COLS,
        "cat_cols": CAT_COLS,
        "cat_cardinalities": [len(vocabs[c]) for c in CAT_COLS],
    }
    with open(os.path.join(out_b_dir, "schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"[DUMP] bucket {bucket:03d}: rows={len(df)} -> {out_b_dir}")

    del df
    gc.collect()


def encode_buckets_to_tensors() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    vocabs = build_vocab_all_buckets()
    with open(os.path.join(OUT_DIR, "vocabs.json"), "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False)

    for b in range(NUM_BUCKETS):
        encode_and_dump(b, vocabs)

    dataset_schema = {
        "NUM_BUCKETS": NUM_BUCKETS,
        "NUM_COLS": NUM_COLS,
        "CAT_COLS": CAT_COLS,
        "cat_cardinalities": [len(vocabs[c]) for c in CAT_COLS],
        "missing_token": MISSING_TOKEN,
    }
    with open(os.path.join(OUT_DIR, "dataset_schema.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_schema, f, ensure_ascii=False, indent=2)

    print("[DONE] tab tensor v1 generated at:", OUT_DIR)


def _has_any_raw_tables() -> bool:
    if not os.path.isdir(IN_TAB_DIR):
        return False
    try:
        from config import TAB_CFG  # noqa: F401
    except Exception:
        return False
    for name in os.listdir(IN_TAB_DIR):
        if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
            return True
    return False


def _has_any_bucket_tables() -> bool:
    if not os.path.isdir(IN_TAB_BUCKET_DIR):
        return False
    for t in TABLES:
        d = os.path.join(IN_TAB_BUCKET_DIR, t)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.startswith("bucket_") and fn.endswith(".csv"):
                return True
    return False


def main() -> None:
    stage = os.environ.get("TAB_STAGE", "auto").strip().lower()

    want_prune = stage in {"auto", "prune", "both"}
    want_encode = stage in {"auto", "encode", "both"}

    can_prune = _has_any_raw_tables()
    can_encode = _has_any_bucket_tables()

    if stage == "prune":
        prune_tabular()
        return
    if stage == "encode":
        encode_buckets_to_tensors()
        return
    if stage == "both":
        prune_tabular()
        encode_buckets_to_tensors()
        return

    if want_prune and can_prune and want_encode and can_encode:
        prune_tabular()
        encode_buckets_to_tensors()
        return

    if want_prune and can_prune:
        prune_tabular()
        return

    if want_encode and can_encode:
        encode_buckets_to_tensors()
        return

    raise RuntimeError(
        "TAB_STAGE=auto but cannot determine runnable stage. "
        f"Check paths: IN_TAB_DIR={IN_TAB_DIR}, IN_TAB_BUCKET_DIR={IN_TAB_BUCKET_DIR}, "
        "and ensure config.TAB_CFG exists for prune."
    )


if __name__ == "__main__":
    main()



