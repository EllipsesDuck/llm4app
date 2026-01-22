import os
import glob
import json
import gc
import csv
from typing import Dict, List, Optional

import pandas as pd

from config import TAB_CFG, TS_CFG


ROOT = r"E:/NUS/data/perdata/train_text_all_samples"

IN_TAB_DIR = os.path.join(ROOT, "tabular_pruned")
IN_TS_DIR  = os.path.join(ROOT, "timeseries_merged_pruned")

OUT_TAB_DIR = os.path.join(ROOT, "tabular_clean_v1")
OUT_TS_DIR  = os.path.join(ROOT, "timeseries_clean_v1")

os.makedirs(OUT_TAB_DIR, exist_ok=True)
os.makedirs(OUT_TS_DIR, exist_ok=True)


CHUNKSIZE = 200_000

NUM_BUCKETS = 256

DROP_BAD_SAMPLE_ID = True      
WRITE_INDEX_FILES = True       
GC_EVERY_N_CHUNKS = 5


CLEAR_OLD_BUCKETS_FOR_TABLES = {"prescriptions"}

def find_table_file(dir_path: str, name: str) -> str:
    cands = []
    for ext in ["csv", "csv.gz", "tsv", "tsv.gz"]:
        cands += glob.glob(os.path.join(dir_path, f"{name}.{ext}"))
    if not cands:
        cands = glob.glob(os.path.join(dir_path, f"{name}.*"))
    if not cands:
        raise FileNotFoundError(f"Cannot find file for table={name} under {dir_path}")
    cands = sorted(cands, key=lambda x: (len(os.path.basename(x)), x))
    return cands[0]

def ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def append_csv(df: pd.DataFrame, out_path: str):
    ensure_dir(out_path)
    header = (not os.path.exists(out_path)) or (os.path.getsize(out_path) == 0)
    df.to_csv(out_path, mode="a", index=False, header=header)

def sample_id_to_int(s: pd.Series) -> pd.Series:
    """
    sample_id expected format: s_000123
    return int id, invalid -> -1
    """
    s = s.astype("string")
    digits = s.str.replace("s_", "", regex=False)
    num = pd.to_numeric(digits, errors="coerce")
    num = num.fillna(-1).astype("int64")
    return num

def add_bucket_col(df: pd.DataFrame, num_buckets: int) -> pd.DataFrame:
    sid_int = sample_id_to_int(df["sample_id"])
    df["_sid_int"] = sid_int
    if DROP_BAD_SAMPLE_ID:
        df = df[df["_sid_int"] >= 0]
    df["_bucket"] = (df["_sid_int"] % num_buckets).astype("int32")
    return df

def _strip_obj_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()
    return df

def _fill_missing_str(df: pd.DataFrame, cols: List[str], fill="__MISSING__"):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
            df[c] = df[c].fillna(fill)
            df[c] = df[c].replace("", fill)
    return df

def safe_cast_chunk(df: pd.DataFrame, dtype_map: Optional[dict], parse_dates: Optional[list]):
    # parse dates safely
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    # cast dtypes safely
    if dtype_map:
        for c, dt in dtype_map.items():
            if c not in df.columns:
                continue
            if dt == "category":
                df[c] = df[c].astype("string")
            else:
                if "int" in dt or "float" in dt:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                try:
                    df[c] = df[c].astype(dt)
                except Exception:
                    pass
    return df

def basic_dedup(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def clean_tab_chunk(table: str, df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise ValueError(f"[{table}] missing sample_id in input")

    # strip string columns
    str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "string"]
    df = _strip_obj_cols(df, str_cols)

    # fill missing for known categorical-like columns
    maybe_cat = [
        "gender", "anchor_year_group", "admission_type", "admission_location",
        "insurance", "language", "marital_status", "first_careunit", "eventtype",
        "careunit", "drug_type", "route", "drug", "icd_code"
    ]
    df = _fill_missing_str(df, [c for c in maybe_cat if c in df.columns])

    # small numeric sanity
    if table == "demographics" and "anchor_age" in df.columns:
        df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce")
        df["anchor_age"] = df["anchor_age"].clip(lower=0, upper=120)

    # time sanity: swap if reversed
    if table in ("icustays", "transfers"):
        if "intime" in df.columns and "outtime" in df.columns:
            bad = df["intime"].notna() & df["outtime"].notna() & (df["intime"] > df["outtime"])
            if bad.any():
                tmp = df.loc[bad, "intime"].copy()
                df.loc[bad, "intime"] = df.loc[bad, "outtime"]
                df.loc[bad, "outtime"] = tmp

    df = df[df["sample_id"].notna()]
    df = basic_dedup(df)
    return df

def clean_ts_chunk(table: str, df: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in df.columns:
        raise ValueError(f"[{table}] missing sample_id in input")

    # strip string columns
    str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "string"]
    df = _strip_obj_cols(df, str_cols)

    # fill missing for uom/flag/locationcategory
    maybe_cat = ["valueuom", "amountuom", "rateuom", "flag", "locationcategory"]
    df = _fill_missing_str(df, [c for c in maybe_cat if c in df.columns])

    # numeric cleaning for "value" in outputevents/procedureevents
    if table in ("outputevents", "procedureevents") and "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[df["sample_id"].notna()]

    # drop rows with all time null (very often useless)
    time_cols = [c for c in ["charttime", "starttime", "endtime"] if c in df.columns]
    if time_cols:
        keep = None
        for c in time_cols:
            keep = df[c].notna() if keep is None else (keep | df[c].notna())
        df = df[keep]

    # itemid missing -> drop
    if "itemid" in df.columns:
        df = df[df["itemid"].notna()]

    df = basic_dedup(df)
    return df

def make_reader(in_fp: str, usecols, chunksize: int, table_name: str):
    """
    C engine first (fast). If ParserError/OOM-like issues, fallback to python engine.
    Always skip bad lines to avoid hard crash on rare corrupted rows.
    """
    if table_name == "prescriptions":
        try:
            return pd.read_csv(
                in_fp,
                usecols=usecols,
                chunksize=min(chunksize, 200_000),
                low_memory=True,
                engine="c",
                on_bad_lines="skip",
            )
        except Exception:
            # fallback python
            return pd.read_csv(
                in_fp,
                usecols=usecols,
                chunksize=min(chunksize, 100_000),
                low_memory=True,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_NONE,
                escapechar="\\",
            )

    try:
        return pd.read_csv(
            in_fp,
            usecols=usecols,
            chunksize=chunksize,
            low_memory=True,
            engine="c",
            on_bad_lines="skip",
        )
    except Exception:
        return pd.read_csv(
            in_fp,
            usecols=usecols,
            chunksize=min(chunksize, 200_000),
            low_memory=True,
            engine="python",
            on_bad_lines="skip",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
        )

def write_bucketed(table: str, df: pd.DataFrame, out_root: str, num_buckets: int) -> Dict[int, int]:
    """
    write df into out_root/{table}/bucket_XXX.csv
    return {bucket: rows_written}
    """
    df = add_bucket_col(df, num_buckets=num_buckets)
    if len(df) == 0:
        return {}

    counts = {}
    for b, g in df.groupby("_bucket", sort=False):
        out_path = os.path.join(out_root, table, f"bucket_{int(b):03d}.csv")
        g = g.drop(columns=["_bucket", "_sid_int"], errors="ignore")
        append_csv(g, out_path)
        counts[int(b)] = counts.get(int(b), 0) + len(g)
    return counts

def update_stats(stats: dict, df: pd.DataFrame, table: str):
    stats.setdefault(table, {})
    s = stats[table]
    s["rows"] = s.get("rows", 0) + len(df)
    for c in df.columns:
        if c not in s:
            s[c] = {"nan": 0, "total": 0}
        s[c]["nan"] += int(df[c].isna().sum())
        s[c]["total"] += int(len(df))
    return stats

def finalize_stats(stats: dict):
    out = {}
    for table, d in stats.items():
        out[table] = {"rows": d.get("rows", 0), "missing": {}}
        for c, vv in d.items():
            if c == "rows":
                continue
            nan = vv["nan"]
            total = vv["total"]
            out[table]["missing"][c] = float(nan) / float(total) if total else 0.0
    return out

def maybe_clear_old_buckets(out_dir: str, table: str):
    if table not in CLEAR_OLD_BUCKETS_FOR_TABLES:
        return
    out_table_dir = os.path.join(out_dir, table)
    if not os.path.exists(out_table_dir):
        return
    olds = glob.glob(os.path.join(out_table_dir, "bucket_*.csv"))
    if olds:
        print(f"[CLEAN] remove old buckets for table={table}, count={len(olds)}")
        for p in olds:
            try:
                os.remove(p)
            except Exception:
                pass

def process_one_group(group_name: str, cfg_map: dict, in_dir: str, out_dir: str, cleaner_fn):
    stats = {}
    manifest = {"group": group_name, "num_buckets": NUM_BUCKETS, "tables": {}}

    for table, cfg in cfg_map.items():
        in_fp = find_table_file(in_dir, table)
        print(f"\n[{group_name}] table={table} file={os.path.basename(in_fp)}")

        usecols = cfg.get("usecols", None)
        dtype_map = cfg.get("dtype", None)
        parse_dates = cfg.get("parse_dates", None)

        out_table_dir = os.path.join(out_dir, table)
        os.makedirs(out_table_dir, exist_ok=True)

        maybe_clear_old_buckets(out_dir, table)

        bucket_counts_total = {}

        reader = make_reader(in_fp, usecols, CHUNKSIZE, table)

        chunk_i = 0
        for chunk in reader:
            chunk_i += 1

            # type cast + note: sample_id 不 cast
            chunk = safe_cast_chunk(chunk, dtype_map, parse_dates)

            # clean
            chunk = cleaner_fn(table, chunk)

            # stats
            update_stats(stats, chunk, table)

            # bucket write
            bucket_counts = write_bucketed(table, chunk, out_dir, NUM_BUCKETS)
            for b, n in bucket_counts.items():
                bucket_counts_total[b] = bucket_counts_total.get(b, 0) + n

            del chunk
            if chunk_i % GC_EVERY_N_CHUNKS == 0:
                gc.collect()

            print(f"  - chunk {chunk_i} done")

        manifest["tables"][table] = {
            "input_file": in_fp,
            "out_dir": out_table_dir,
            "bucket_counts": bucket_counts_total,
        }

        print(f"[{group_name}] table={table} done. buckets_written={len(bucket_counts_total)}")

    if WRITE_INDEX_FILES:
        stats_final = finalize_stats(stats)
        with open(os.path.join(out_dir, f"{group_name}_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats_final, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_dir, f"{group_name}_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"\n[{group_name}] saved stats/manifest to: {out_dir}")

def main():
    process_one_group(
        group_name="timeseries_clean_v1",
        cfg_map=TS_CFG,
        in_dir=IN_TS_DIR,
        out_dir=OUT_TS_DIR,
        cleaner_fn=clean_ts_chunk
    )

    print("\n[DONE] clean v1 finished.")
    print("TAB out:", OUT_TAB_DIR)
    print("TS  out:", OUT_TS_DIR)

if __name__ == "__main__":
    main()

