import gc
import glob
import json
import os
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config import TS_CFG


ROOT = r"E:/NUS/data/perdata/train_text_all_samples"

IN_TS_DIR = os.path.join(ROOT, "timeseries_merged")
OUT_TS_PRUNED_DIR = os.path.join(ROOT, "timeseries_merged_pruned")

IN_TS_BUCKET_ROOT = os.path.join(ROOT, "timeseries_clean_v1")
OUT_TS_FIXED_DIR = os.path.join(ROOT, "timeseries_tensor_v1")
OUT_TS_RAGGED_DIR = os.path.join(ROOT, "timeseries_ragged_v1")

NUM_BUCKETS = 256
CHUNKSIZE_DEFAULT = 1_000_000
CHUNKSIZE = 200_000

L = 256
HASH_V = 100_000

SAVE_FLOAT16 = False
SKIP_IF_EXISTS = True
USE_ENDTIME_AS_FALLBACK = True

SORT_WITHIN_SAMPLE = True
COMPUTE_TREF = False


TABLES = ["chartevents", "labevents", "inputevents", "outputevents", "procedureevents"]
SRC_ID = {t: i for i, t in enumerate(TABLES)}

TABLE_SCHEMA = {
    "chartevents": {"time": ["charttime"], "value": ["valuenum"]},
    "labevents": {"time": ["charttime"], "value": ["valuenum"]},
    "inputevents": {"time": ["starttime", "endtime"], "value": ["amount", "rate"]},
    "outputevents": {"time": ["charttime"], "value": ["value"]},
    "procedureevents": {"time": ["starttime", "endtime"], "value": ["value"]},
}


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


def prune_ts() -> None:
    os.makedirs(OUT_TS_PRUNED_DIR, exist_ok=True)

    for name, cfg in TS_CFG.items():
        in_fp = find_table_file(IN_TS_DIR, name)
        out_fp = os.path.join(OUT_TS_PRUNED_DIR, f"{name}.csv")

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

    print(f"[DONE] pruned ts saved to: {OUT_TS_PRUNED_DIR}")


def _read_csv_chunks(path: str, usecols: Optional[List[str]], chunksize: int):
    kwargs = dict(chunksize=chunksize, low_memory=True)
    if usecols is not None:
        kwargs["usecols"] = usecols
    try:
        return pd.read_csv(path, on_bad_lines="skip", **kwargs)
    except TypeError:
        return pd.read_csv(path, error_bad_lines=False, warn_bad_lines=True, **kwargs)


def find_bucket_file(table: str, bucket_id: int) -> Optional[str]:
    p = os.path.join(IN_TS_BUCKET_ROOT, table, f"bucket_{bucket_id:03d}.csv")
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    cands = glob.glob(os.path.join(IN_TS_BUCKET_ROOT, table, f"bucket_{bucket_id:03d}.*"))
    if not cands:
        return None
    cands = sorted(cands, key=lambda x: (len(os.path.basename(x)), x))
    return cands[0]


def stable_hash_to_id(s: str, mod: int) -> int:
    h = zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF
    return (h % mod) + 1


def to_datetime_ns(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.view("int64").to_numpy()


def pick_first_existing(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def compute_tref_for_bucket_fixedL(bucket_id: int) -> Dict[str, int]:
    tref: Dict[str, int] = {}

    for table in TABLES:
        fp = find_bucket_file(table, bucket_id)
        if fp is None:
            continue

        time_cols = TABLE_SCHEMA[table]["time"]
        usecols = ["sample_id"] + time_cols

        try:
            reader = _read_csv_chunks(fp, usecols=usecols, chunksize=CHUNKSIZE)
        except ValueError:
            reader = _read_csv_chunks(fp, usecols=None, chunksize=CHUNKSIZE)

        for chunk in reader:
            if "sample_id" not in chunk.columns:
                continue

            tcol = pick_first_existing(chunk, time_cols)
            if tcol is None:
                continue

            t_ns = to_datetime_ns(chunk[tcol])
            ok = t_ns > 0
            if not np.any(ok):
                continue

            sid = chunk.loc[ok, "sample_id"].astype("string").to_numpy()
            t_ok = t_ns[ok]

            tmp = pd.DataFrame({"sid": sid, "t": t_ok})
            gmax = tmp.groupby("sid", sort=False)["t"].max()

            for s, tmax in gmax.items():
                prev = tref.get(s)
                if prev is None or int(tmax) > prev:
                    tref[s] = int(tmax)

            del chunk, tmp, gmax
        gc.collect()

    return tref


def build_fixedL_for_bucket(bucket_id: int, tref: Dict[str, int]) -> Dict[str, torch.Tensor]:
    events: Dict[str, List[Tuple[int, int, int, float]]] = {}

    for table in TABLES:
        fp = find_bucket_file(table, bucket_id)
        if fp is None:
            continue

        schema = TABLE_SCHEMA[table]
        time_cands = schema["time"]
        value_cands = schema["value"]

        usecols = ["sample_id", "itemid"] + time_cands + value_cands
        usecols = list(dict.fromkeys(usecols))

        try:
            reader = _read_csv_chunks(fp, usecols=usecols, chunksize=CHUNKSIZE)
        except ValueError:
            reader = _read_csv_chunks(fp, usecols=None, chunksize=CHUNKSIZE)

        src_token = SRC_ID[table] + 1

        for chunk in reader:
            if "sample_id" not in chunk.columns or "itemid" not in chunk.columns:
                continue

            tcol = pick_first_existing(chunk, time_cands)
            if tcol is None and USE_ENDTIME_AS_FALLBACK:
                tcol = pick_first_existing(chunk, ["endtime"])
            if tcol is None:
                continue

            vcol = pick_first_existing(chunk, value_cands)
            if vcol is None:
                continue

            t_ns = to_datetime_ns(chunk[tcol])
            v = pd.to_numeric(chunk[vcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            sid = chunk["sample_id"].astype("string").to_numpy()
            item = pd.to_numeric(chunk["itemid"], errors="coerce").to_numpy(dtype=np.float64, copy=False)

            ok = (t_ns > 0) & np.isfinite(v) & np.isfinite(item)
            if not np.any(ok):
                continue

            sid_ok = sid[ok]
            t_ok = t_ns[ok].astype(np.int64, copy=False)
            v_ok = v[ok].astype(np.float32, copy=False)
            item_ok = item[ok].astype(np.int64, copy=False)

            for s, tt, vv, it in zip(sid_ok, t_ok, v_ok, item_ok):
                if s not in tref:
                    continue
                item_id = stable_hash_to_id(f"{table}:{int(it)}", HASH_V)
                events.setdefault(s, []).append((int(tt), int(src_token), int(item_id), float(vv)))

            del chunk
        gc.collect()

    sample_ids = sorted(events.keys())
    N = len(sample_ids)

    item_mat = torch.zeros((N, L), dtype=torch.long)
    src_mat = torch.zeros((N, L), dtype=torch.long)
    value_mat = torch.zeros((N, L), dtype=torch.float16 if SAVE_FLOAT16 else torch.float32)
    dt_mat = torch.zeros((N, L), dtype=torch.float16 if SAVE_FLOAT16 else torch.float32)
    mask = torch.zeros((N, L), dtype=torch.bool)

    for i, sid in enumerate(sample_ids):
        ev = events[sid]
        ev.sort(key=lambda x: x[0])
        if len(ev) > L:
            ev = ev[-L:]

        t_ref = tref[sid]
        m = len(ev)
        if m == 0:
            continue

        start = L - m
        for j, (t_ns, src_id, item_id, val) in enumerate(ev):
            pos = start + j
            if pos >= L:
                break
            item_mat[i, pos] = item_id
            src_mat[i, pos] = src_id
            value_mat[i, pos] = val
            dt_mat[i, pos] = (t_ns - t_ref) / (3600.0 * 1e9)
            mask[i, pos] = True

    out = {
        "sample_id": sample_ids,
        "item": item_mat,
        "src": src_mat,
        "value": value_mat,
        "dt": dt_mat,
        "mask": mask,
        "meta": {
            "bucket_id": bucket_id,
            "L": L,
            "HASH_V": HASH_V,
            "tables": TABLES,
            "src_id_map": {k: int(v + 1) for k, v in SRC_ID.items()},
            "pad_item": 0,
            "pad_src": 0,
            "format": "fixedL",
        },
    }
    return out


def export_fixedL() -> None:
    os.makedirs(OUT_TS_FIXED_DIR, exist_ok=True)

    stats: Dict[str, dict] = {}
    for b in range(NUM_BUCKETS):
        out_pt = os.path.join(OUT_TS_FIXED_DIR, f"bucket_{b:03d}.pt")
        out_json = os.path.join(OUT_TS_FIXED_DIR, f"bucket_{b:03d}.json")

        if SKIP_IF_EXISTS and os.path.exists(out_pt):
            print(f"[SKIP] bucket {b:03d} exists: {out_pt}")
            continue

        print(f"\n[BUCKET {b:03d}] pass1: compute t_ref ...")
        tref = compute_tref_for_bucket_fixedL(b)
        print(f"[BUCKET {b:03d}] t_ref samples: {len(tref)}")

        if len(tref) == 0:
            print(f"[BUCKET {b:03d}] empty, skip.")
            continue

        print(f"[BUCKET {b:03d}] pass2: build fixed-L tensors ...")
        out = build_fixedL_for_bucket(b, tref)

        torch.save(out, out_pt)

        N = len(out["sample_id"])
        nnz = int(out["mask"].sum().item())
        stats_b = {
            "bucket": b,
            "samples": N,
            "events_kept": nnz,
            "avg_events_per_sample": float(nnz) / float(N) if N else 0.0,
            "L": L,
            "HASH_V": HASH_V,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats_b, f, ensure_ascii=False, indent=2)

        stats[str(b)] = stats_b
        print(f"[BUCKET {b:03d}] saved: {out_pt}")
        print(f"  - samples={N}, events_kept={nnz}, avg={stats_b['avg_events_per_sample']:.2f}")

        del tref, out
        gc.collect()

    with open(os.path.join(OUT_TS_FIXED_DIR, "all_buckets_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n[DONE] all buckets processed.")
    print("OUT:", OUT_TS_FIXED_DIR)


def count_events_for_bucket(bucket_id: int) -> Tuple[List[str], Dict[str, int]]:
    counts: Dict[str, int] = {}

    for table in TABLES:
        fp = find_bucket_file(table, bucket_id)
        if fp is None:
            continue

        schema = TABLE_SCHEMA[table]
        time_cands = schema["time"]
        value_cands = schema["value"]

        usecols = ["sample_id", "itemid"] + time_cands + value_cands
        usecols = list(dict.fromkeys(usecols))

        try:
            reader = _read_csv_chunks(fp, usecols=usecols, chunksize=CHUNKSIZE)
        except ValueError:
            reader = _read_csv_chunks(fp, usecols=None, chunksize=CHUNKSIZE)

        for chunk in reader:
            if "sample_id" not in chunk.columns or "itemid" not in chunk.columns:
                continue

            tcol = pick_first_existing(chunk, time_cands)
            vcol = pick_first_existing(chunk, value_cands)
            if tcol is None or vcol is None:
                continue

            t_ns = to_datetime_ns(chunk[tcol])
            v = pd.to_numeric(chunk[vcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            item = pd.to_numeric(chunk["itemid"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            sid = chunk["sample_id"].astype("string").to_numpy()

            ok = (t_ns > 0) & np.isfinite(v) & np.isfinite(item)
            if not np.any(ok):
                continue

            sid_ok = sid[ok]
            vc = pd.Series(sid_ok).value_counts()
            for s, c in vc.items():
                counts[s] = counts.get(s, 0) + int(c)

            del chunk, vc
        gc.collect()

    sample_ids = sorted(counts.keys())
    return sample_ids, counts


def compute_tref_for_bucket_ragged(bucket_id: int, sample_set: set) -> Dict[str, int]:
    tref: Dict[str, int] = {}
    for table in TABLES:
        fp = find_bucket_file(table, bucket_id)
        if fp is None:
            continue

        time_cands = TABLE_SCHEMA[table]["time"]
        usecols = ["sample_id"] + time_cands

        try:
            reader = _read_csv_chunks(fp, usecols=usecols, chunksize=CHUNKSIZE)
        except ValueError:
            reader = _read_csv_chunks(fp, usecols=None, chunksize=CHUNKSIZE)

        for chunk in reader:
            if "sample_id" not in chunk.columns:
                continue
            tcol = pick_first_existing(chunk, time_cands)
            if tcol is None:
                continue

            t_ns = to_datetime_ns(chunk[tcol])
            ok = t_ns > 0
            if not np.any(ok):
                continue

            sid = chunk.loc[ok, "sample_id"].astype("string").to_numpy()
            t_ok = t_ns[ok].astype(np.int64, copy=False)

            tmp = pd.DataFrame({"sid": sid, "t": t_ok})
            tmp = tmp[tmp["sid"].isin(sample_set)]
            if len(tmp) == 0:
                continue

            gmax = tmp.groupby("sid", sort=False)["t"].max()
            for s, tmax in gmax.items():
                prev = tref.get(s)
                if prev is None or int(tmax) > prev:
                    tref[s] = int(tmax)

            del chunk, tmp, gmax
        gc.collect()

    return tref


def build_ragged_for_bucket(
    bucket_id: int,
    sample_ids: List[str],
    counts: Dict[str, int],
    tref: Optional[Dict[str, int]] = None,
):
    N = len(sample_ids)
    if N == 0:
        return None

    indptr = np.zeros((N + 1,), dtype=np.int64)
    for i, sid in enumerate(sample_ids):
        indptr[i + 1] = indptr[i] + int(counts[sid])
    E = int(indptr[-1])

    t_arr = np.zeros((E,), dtype=np.int64)
    src_arr = np.zeros((E,), dtype=np.int64)
    item_arr = np.zeros((E,), dtype=np.int64)
    val_arr = np.zeros((E,), dtype=np.float16 if SAVE_FLOAT16 else np.float32)

    write_ptr = indptr[:-1].copy()
    sid2idx = {sid: i for i, sid in enumerate(sample_ids)}

    for table in TABLES:
        fp = find_bucket_file(table, bucket_id)
        if fp is None:
            continue

        schema = TABLE_SCHEMA[table]
        time_cands = schema["time"]
        value_cands = schema["value"]

        usecols = ["sample_id", "itemid"] + time_cands + value_cands
        usecols = list(dict.fromkeys(usecols))

        try:
            reader = _read_csv_chunks(fp, usecols=usecols, chunksize=CHUNKSIZE)
        except ValueError:
            reader = _read_csv_chunks(fp, usecols=None, chunksize=CHUNKSIZE)

        src_token = SRC_ID[table] + 1

        for chunk in reader:
            if "sample_id" not in chunk.columns or "itemid" not in chunk.columns:
                continue

            tcol = pick_first_existing(chunk, time_cands)
            vcol = pick_first_existing(chunk, value_cands)
            if tcol is None or vcol is None:
                continue

            t_ns = to_datetime_ns(chunk[tcol])
            v = pd.to_numeric(chunk[vcol], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            item = pd.to_numeric(chunk["itemid"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            sid = chunk["sample_id"].astype("string").to_numpy()

            ok = (t_ns > 0) & np.isfinite(v) & np.isfinite(item)
            if not np.any(ok):
                continue

            sid_ok = sid[ok]
            t_ok = t_ns[ok].astype(np.int64, copy=False)
            v_ok = v[ok].astype(np.float32, copy=False)
            item_ok = item[ok].astype(np.int64, copy=False)

            for s, tt, vv, it in zip(sid_ok, t_ok, v_ok, item_ok):
                idx = sid2idx.get(s)
                if idx is None:
                    continue
                pos = write_ptr[idx]
                if pos >= indptr[idx + 1]:
                    continue
                write_ptr[idx] += 1

                t_arr[pos] = int(tt)
                src_arr[pos] = int(src_token)
                item_arr[pos] = int(stable_hash_to_id(f"{table}:{int(it)}", HASH_V))
                val_arr[pos] = vv

            del chunk
        gc.collect()

    if SORT_WITHIN_SAMPLE:
        for i in range(N):
            a = int(indptr[i])
            b = int(indptr[i + 1])
            if b - a <= 1:
                continue
            order = np.argsort(t_arr[a:b], kind="mergesort")
            if np.all(order == np.arange(b - a)):
                continue
            t_arr[a:b] = t_arr[a:b][order]
            src_arr[a:b] = src_arr[a:b][order]
            item_arr[a:b] = item_arr[a:b][order]
            val_arr[a:b] = val_arr[a:b][order]

    out = {
        "sample_id": sample_ids,
        "indptr": torch.from_numpy(indptr).long(),
        "t_ns": torch.from_numpy(t_arr).long(),
        "src": torch.from_numpy(src_arr).long(),
        "item": torch.from_numpy(item_arr).long(),
        "value": torch.from_numpy(val_arr).half() if SAVE_FLOAT16 else torch.from_numpy(val_arr).float(),
        "meta": {
            "bucket_id": bucket_id,
            "HASH_V": HASH_V,
            "tables": TABLES,
            "src_id_map": {k: int(v + 1) for k, v in SRC_ID.items()},
            "format": "ragged_csr",
            "sorted_within_sample": bool(SORT_WITHIN_SAMPLE),
        },
    }

    if tref is not None:
        tref_arr = np.array([int(tref.get(s, 0)) for s in sample_ids], dtype=np.int64)
        out["t_ref_ns"] = torch.from_numpy(tref_arr).long()

    return out


def export_ragged() -> None:
    os.makedirs(OUT_TS_RAGGED_DIR, exist_ok=True)

    all_stats: Dict[str, dict] = {}
    for b in range(NUM_BUCKETS):
        out_pt = os.path.join(OUT_TS_RAGGED_DIR, f"bucket_{b:03d}.pt")
        out_json = os.path.join(OUT_TS_RAGGED_DIR, f"bucket_{b:03d}.json")

        if SKIP_IF_EXISTS and os.path.exists(out_pt):
            print(f"[SKIP] bucket {b:03d} exists: {out_pt}")
            continue

        print(f"\n[BUCKET {b:03d}] passA: count events ...")
        sample_ids, counts = count_events_for_bucket(b)
        N = len(sample_ids)
        E = int(sum(counts.values()))
        print(f"[BUCKET {b:03d}] samples={N} total_events={E}")

        if N == 0 or E == 0:
            print(f"[BUCKET {b:03d}] empty, skip.")
            continue

        tref = None
        if COMPUTE_TREF:
            print(f"[BUCKET {b:03d}] passB: compute t_ref ...")
            tref = compute_tref_for_bucket_ragged(b, sample_set=set(sample_ids))
            print(f"[BUCKET {b:03d}] t_ref computed for {len(tref)} samples")

        print(f"[BUCKET {b:03d}] passC: build ragged tensors ...")
        out = build_ragged_for_bucket(b, sample_ids, counts, tref=tref)

        torch.save(out, out_pt)

        stats_b = {
            "bucket": b,
            "samples": int(N),
            "events": int(E),
            "avg_events_per_sample": float(E) / float(N),
            "HASH_V": int(HASH_V),
            "sorted_within_sample": bool(SORT_WITHIN_SAMPLE),
            "compute_tref": bool(COMPUTE_TREF),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats_b, f, ensure_ascii=False, indent=2)

        all_stats[str(b)] = stats_b
        print(f"[BUCKET {b:03d}] saved: {out_pt}")
        print(f"  - avg_events_per_sample={stats_b['avg_events_per_sample']:.2f}")

        del out, sample_ids, counts, tref
        gc.collect()

    with open(os.path.join(OUT_TS_RAGGED_DIR, "all_buckets_stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print("\n[DONE] ragged export finished.")
    print("OUT:", OUT_TS_RAGGED_DIR)


def _has_any_merged_ts() -> bool:
    if not os.path.isdir(IN_TS_DIR):
        return False
    for fn in os.listdir(IN_TS_DIR):
        if fn.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz")):
            return True
    return False


def _has_any_ts_buckets() -> bool:
    if not os.path.isdir(IN_TS_BUCKET_ROOT):
        return False
    for t in TABLES:
        d = os.path.join(IN_TS_BUCKET_ROOT, t)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.startswith("bucket_") and fn.endswith(".csv"):
                return True
    return False


def main() -> None:
    stage = os.environ.get("TS_STAGE", "auto").strip().lower()
    export_kind = os.environ.get("TS_EXPORT", "ragged").strip().lower()

    can_prune = _has_any_merged_ts()
    can_export = _has_any_ts_buckets()

    if stage == "prune":
        prune_ts()
        return

    if stage == "fixed":
        export_fixedL()
        return

    if stage == "ragged":
        export_ragged()
        return

    if stage == "both":
        if can_prune:
            prune_ts()
        if export_kind == "fixed":
            export_fixedL()
        else:
            export_ragged()
        return

    if stage != "auto":
        raise ValueError(f"Unsupported TS_STAGE={stage}")

    if can_prune:
        prune_ts()

    if not can_export:
        return

    if export_kind == "fixed":
        export_fixedL()
    else:
        export_ragged()


if __name__ == "__main__":
    main()

