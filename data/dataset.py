import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from config import TAB_CFG, TS_CFG


def load_text_table(TEXT_PATH: str):
    if not os.path.exists(TEXT_PATH):
        print("[TEXT][WARN] text.csv not found")
        return None, {}, {}

    df_text = pd.read_csv(TEXT_PATH, low_memory=False)

    if "subject_id" in df_text.columns:
        df_text["subject_id"] = pd.to_numeric(df_text["subject_id"], errors="coerce")
        df_text = df_text[df_text["subject_id"].notna()].copy()
        df_text["subject_id"] = df_text["subject_id"].astype("int32")

    if "cxr_time" in df_text.columns:
        df_text["cxr_time"] = pd.to_datetime(df_text["cxr_time"], errors="coerce")

    if "text" in df_text.columns:
        df_text["text"] = df_text["text"].fillna("").astype(str)
    else:
        df_text["text"] = ""

    text_map_dicom: Dict[str, str] = {}
    if "dicom_id" in df_text.columns:
        df_text["dicom_id"] = df_text["dicom_id"].astype(str)
        text_map_dicom = (
            df_text.groupby("dicom_id")["text"]
            .apply(lambda s: "\n".join([t.strip() for t in s if isinstance(t, str) and t.strip()]))
            .to_dict()
        )

    text_map_subject: Dict[int, Dict[str, Any]] = {}
    if "subject_id" in df_text.columns and "cxr_time" in df_text.columns:
        df_valid = df_text[df_text["cxr_time"].notna()]
        for sid, g in df_valid.groupby("subject_id", sort=False):
            gg = g.sort_values("cxr_time")
            text_map_subject[int(sid)] = {
                "times": gg["cxr_time"].to_numpy(dtype="datetime64[ns]"),
                "texts": gg["text"].to_numpy(dtype=object),
            }

    print(
        f"[TEXT] rows={len(df_text):,} dicom_map={len(text_map_dicom):,} "
        f"subject_map={len(text_map_subject):,}"
    )
    return df_text, text_map_dicom, text_map_subject


def query_text(
    text_map_dicom,
    text_map_subject,
    subject_id: int,
    dicom_id,
    cxr_time: pd.Timestamp,
    tol_minutes=5,
):
    if dicom_id is not None:
        did = str(dicom_id)
        if did in text_map_dicom:
            return text_map_dicom[did]

    if pd.isna(cxr_time):
        return ""

    rec = text_map_subject.get(int(subject_id))
    if rec is None:
        return ""

    times = rec["times"]
    texts = rec["texts"]
    if len(times) == 0:
        return ""

    t = np.datetime64(cxr_time.to_datetime64())
    tol = np.timedelta64(int(tol_minutes * 60), "s")
    left = np.searchsorted(times, t - tol, side="left")
    right = np.searchsorted(times, t + tol, side="right")
    if right <= left:
        return ""

    cand = [str(x).strip() for x in texts[left:right] if isinstance(x, str) and x.strip()]
    return "\n".join(cand)


def read_csv_light(path, usecols=None, dtype=None, parse_dates=None, chunksize=None):
    return pd.read_csv(
        path,
        usecols=usecols,
        dtype=dtype,
        parse_dates=parse_dates,
        low_memory=False,
        chunksize=chunksize,
    )


def build_subject_index(df, key="subject_id"):
    return df.groupby(key, sort=False).indices


def load_tabular_all(TAB_DIR: str):
    tab_df: Dict[str, pd.DataFrame] = {}
    tab_idx: Dict[str, Dict[int, np.ndarray]] = {}

    for name, cfg in TAB_CFG.items():
        path = os.path.join(TAB_DIR, f"{name}.csv")
        if not os.path.exists(path):
            print(f"[TAB][SKIP] {name}.csv not found")
            continue

        df = read_csv_light(
            path,
            usecols=cfg.get("usecols"),
            dtype=cfg.get("dtype"),
            parse_dates=cfg.get("parse_dates"),
            chunksize=None,
        )

        if "subject_id" in df.columns:
            df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
            df = df[df["subject_id"].notna()].copy()
            df["subject_id"] = df["subject_id"].astype("int32")

        tab_df[name] = df
        tab_idx[name] = build_subject_index(df, "subject_id") if "subject_id" in df.columns else {}
        print(f"[TAB] {name}: rows={len(df):,} cols={len(df.columns)}")

    return tab_df, tab_idx


def query_tabular_indices(tab_idx, subject_id: int):
    sid = int(subject_id)
    out: Dict[str, list] = {}
    for name, idx_map in tab_idx.items():
        inds = idx_map.get(sid)
        out[name] = inds.tolist() if inds is not None else []
    return out


def query_tabular(tab_df, tab_idx, subject_id: int):
    sid = int(subject_id)
    out: Dict[str, pd.DataFrame] = {}
    for name, df in tab_df.items():
        idx_map = tab_idx.get(name, {})
        inds = idx_map.get(sid)
        if inds is None:
            out[name] = df.iloc[0:0].copy()
        else:
            out[name] = df.iloc[inds].copy()
    return out


def _ensure_datetime_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def bucketize_timeseries_csv(
    TS_DIR: str,
    OUT_DIR: str,
    num_buckets: int = 512,
    chunksize: int = 1_000_000,
):
    os.makedirs(OUT_DIR, exist_ok=True)

    tables = sorted(list(TS_CFG.keys()))
    for table in tables:
        in_path = os.path.join(TS_DIR, f"{table}.csv")
        if not os.path.exists(in_path):
            print(f"[TS-BUCKET][SKIP] {table}.csv not found")
            continue

        cfg = TS_CFG[table]
        out_table_dir = os.path.join(OUT_DIR, table)
        os.makedirs(out_table_dir, exist_ok=True)

        print(f"\n[TS-BUCKET] table={table} file={table}.csv")

        wrote_header = set()

        reader = pd.read_csv(
            in_path,
            usecols=cfg.get("usecols"),
            dtype=cfg.get("dtype"),
            parse_dates=cfg.get("parse_dates"),
            low_memory=False,
            chunksize=chunksize,
        )

        maybe_time_cols = []
        if cfg.get("parse_dates"):
            maybe_time_cols.extend(list(cfg["parse_dates"]))

        for c in ["charttime", "storetime", "starttime", "endtime", "intime", "outtime"]:
            usecols_cfg = cfg.get("usecols")
            if usecols_cfg is None or c in usecols_cfg:
                if c not in maybe_time_cols:
                    maybe_time_cols.append(c)

        for chunk_id, df in enumerate(reader):
            if "subject_id" not in df.columns:
                print(f"[WARN] {table} has no subject_id, skip this table.")
                break

            df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
            df = df[df["subject_id"].notna()].copy()
            df["subject_id"] = df["subject_id"].astype("int32")

            df = _ensure_datetime_cols(df, maybe_time_cols)

            b = (df["subject_id"].astype("int64") % int(num_buckets)).astype("int32")
            df["_bucket"] = b

            for buck, sub in df.groupby("_bucket", sort=False):
                out_path = os.path.join(out_table_dir, f"bucket_{int(buck):03d}.csv")
                need_header = (out_path not in wrote_header) and (not os.path.exists(out_path))

                sub = sub.drop(columns=["_bucket"])
                sub.to_csv(out_path, mode="a", index=False, header=need_header)

                wrote_header.add(out_path)

            print(f"  chunk {chunk_id}: rows={len(df):,}")

        print(f"[TS-BUCKET] done table={table}")


class BucketCache:
    def __init__(self, max_items=8):
        self.max_items = max_items
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


class TimeseriesBucketReader:
    def __init__(self, bucket_root: str, num_buckets: int = 512, cache_size: int = 16):
        self.bucket_root = bucket_root
        self.num_buckets = num_buckets
        self.cache = BucketCache(max_items=cache_size)

    def _load_bucket(self, table: str, bucket_id: int) -> pd.DataFrame:
        key = (table, bucket_id)
        df = self.cache.get(key)
        if df is not None:
            return df

        path = os.path.join(self.bucket_root, table, f"bucket_{bucket_id:03d}.csv")
        if not os.path.exists(path):
            df = pd.DataFrame()
            self.cache.put(key, df)
            return df

        df = pd.read_csv(path, low_memory=False)

        if "subject_id" in df.columns:
            df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce")
            df = df[df["subject_id"].notna()].copy()
            df["subject_id"] = df["subject_id"].astype("int32")

        for c in ["charttime", "storetime", "starttime", "endtime", "intime", "outtime"]:
            if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = pd.to_datetime(df[c], errors="coerce")

        self.cache.put(key, df)
        return df

    def query_window(self, table: str, subject_id: int, cxr_time, hours: int = 48) -> pd.DataFrame:
        if pd.isna(cxr_time):
            return pd.DataFrame()

        sid = int(subject_id)
        bucket_id = sid % self.num_buckets

        df = self._load_bucket(table, bucket_id)
        if df is None or len(df) == 0 or "subject_id" not in df.columns:
            return pd.DataFrame()

        out = df[df["subject_id"] == sid]
        if len(out) == 0:
            return out

        t0 = pd.to_datetime(cxr_time, errors="coerce") - pd.Timedelta(hours=hours)
        t1 = pd.to_datetime(cxr_time, errors="coerce")
        if pd.isna(t0) or pd.isna(t1):
            return pd.DataFrame()

        if "charttime" in out.columns:
            ct = out["charttime"]
            out = out[ct.notna()]
            return out[(ct >= t0) & (ct <= t1)]

        if "starttime" in out.columns:
            out = out[out["starttime"].notna()]
            st = out["starttime"]

            if "endtime" in out.columns:
                et = out["endtime"]
                et_filled = et.fillna(st)
                return out[(st <= t1) & (et_filled >= t0)]

            return out[(st >= t0) & (st <= t1)]

        return out
