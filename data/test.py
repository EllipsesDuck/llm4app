import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict

from config import TAB_CFG, TS_CFG

from dataset import load_tabular_all, TimeseriesBucketReader, load_text_table, query_text, query_tabular


def iter_all_samples(
    df_index: pd.DataFrame,
    IMG_DIR: str,
    tab_df, tab_idx,
    text_map_dicom, text_map_subject,
    ts_reader: TimeseriesBucketReader,
    ts_tables,
    hours: int = 48,
):
    for row in df_index.itertuples(index=False):
        sample_id = getattr(row, "sample_id")
        subject_id = int(getattr(row, "subject_id"))
        dicom_id = getattr(row, "dicom_id", None)
        cxr_time = pd.to_datetime(getattr(row, "cxr_time"), errors="coerce")

        sample = {
            "sample_id": sample_id,
            "subject_id": subject_id,
            "dicom_id": dicom_id,
            "cxr_time": cxr_time,

            "label_any": getattr(row, "label_any", None),
            "label_multi": getattr(row, "label_multi", None),

            "meta": row._asdict(),
        }

        img_path = os.path.join(IMG_DIR, f"{sample_id}.npy")
        sample["image"] = np.load(img_path) if os.path.exists(img_path) else None

        sample["text"] = query_text(
            text_map_dicom=text_map_dicom,
            text_map_subject=text_map_subject,
            subject_id=subject_id,
            dicom_id=dicom_id,
            cxr_time=cxr_time,
            tol_minutes=5,
        )

        sample["tabular"] = query_tabular(tab_df, tab_idx, subject_id)

        ts_data = {}
        for tname in ts_tables:
            ts_data[tname] = ts_reader.query_window(
                table=tname,
                subject_id=subject_id,
                cxr_time=cxr_time,
                hours=hours,
            )
        sample["timeseries"] = ts_data

        yield sample


def process_one(sample: dict):
    out = {
        "sample_id": sample["sample_id"],
        "ts_lens": {k: len(v) for k, v in sample["timeseries"].items()},
        "text_len": len(sample["text"]),
        "has_image": sample["image"] is not None,
    }
    return out


if __name__ == "__main__":
    ROOT = r"E:/NUS/data/perdata/train_text_samples"
    META_DIR = os.path.join(ROOT, "meta")
    IMG_DIR = os.path.join(ROOT, "images")
    TAB_DIR = os.path.join(ROOT, "tabular")
    TEXT_PATH = os.path.join(ROOT, "text.csv")

    TS_BUCKET_ROOT = r"E:/NUS/data/perdata/timeseries_bucketed"
    SAMPLE_INDEX_PATH = os.path.join(META_DIR, "sample_index.json")

    with open(SAMPLE_INDEX_PATH, "r") as f:
        sample_index = json.load(f)
    df_index = pd.DataFrame(sample_index)
    print("Total samples:", len(df_index))

    # load once
    tab_df, tab_idx = load_tabular_all(TAB_DIR)
    _, text_map_dicom, text_map_subject = load_text_table(TEXT_PATH)

    ts_reader = TimeseriesBucketReader(
        bucket_root=TS_BUCKET_ROOT,
        num_buckets=128,
        cache_size=6,
    )
    ts_tables = list(TS_CFG.keys())

    results = []
    for i, sample in enumerate(iter_all_samples(
        df_index=df_index,
        IMG_DIR=IMG_DIR,
        tab_df=tab_df,
        tab_idx=tab_idx,
        text_map_dicom=text_map_dicom,
        text_map_subject=text_map_subject,
        ts_reader=ts_reader,
        ts_tables=ts_tables,
        hours=48,
    )):
        r = process_one(sample)  
        results.append(r)        

        if (i + 1) % 1000 == 0:
            print(f"[PROGRESS] {i+1:,}/{len(df_index):,}")

    print("done, results:", len(results))

    