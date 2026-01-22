from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch


MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"


def _to_ts(x: Any) -> pd.Timestamp:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    return pd.to_datetime(x, errors="coerce")


def _safe_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return MISSING_TOKEN
    s = str(x).strip()
    return s if s else MISSING_TOKEN


def _select_latest_row_before(
    df: pd.DataFrame, time_col: str, t: pd.Timestamp
) -> Optional[pd.Series]:
    if df is None or len(df) == 0 or pd.isna(t) or time_col not in df.columns:
        return None

    tt = pd.to_datetime(df[time_col], errors="coerce")
    m = tt <= t
    if not m.any():
        return None

    sub = df.loc[m].copy()
    sub["_t"] = pd.to_datetime(sub[time_col], errors="coerce")
    sub = sub.sort_values("_t", ascending=False)
    return sub.iloc[0]


def _select_active_interval_row(
    df: pd.DataFrame,
    intime: str,
    outtime: str,
    t: pd.Timestamp,
) -> Optional[pd.Series]:
    if df is None or len(df) == 0 or pd.isna(t) or intime not in df.columns:
        return None

    tin = pd.to_datetime(df[intime], errors="coerce")
    cond = tin <= t

    if outtime in df.columns:
        tout = pd.to_datetime(df[outtime], errors="coerce")
        cond = cond & (tout.isna() | (tout >= t))

    if not cond.any():
        return None

    sub = df.loc[cond].copy()
    sub["_tin"] = pd.to_datetime(sub[intime], errors="coerce")
    sub = sub.sort_values("_tin", ascending=False)
    return sub.iloc[0]


def _count_window_rows(df: pd.DataFrame, time_col: str, t: pd.Timestamp, hours: int) -> pd.DataFrame:
    if df is None or len(df) == 0 or pd.isna(t) or time_col not in df.columns:
        return df.iloc[0:0] if df is not None else pd.DataFrame()

    tt = pd.to_datetime(df[time_col], errors="coerce")
    t0 = t - pd.Timedelta(hours=hours)
    m = (tt <= t) & (tt >= t0)
    return df.loc[m].copy()


class CategoricalVocab:
    def __init__(self, max_size: int = 50, add_other: bool = True):
        self.max_size = max_size
        self.add_other = add_other
        self.vocab: Dict[str, int] = {}
        self.items: List[str] = []

    def fit_from_values(self, values: List[str]) -> None:
        cnt = Counter(values)
        cnt[MISSING_TOKEN] += 10**9

        most = [k for k, _ in cnt.most_common(self.max_size)]
        if self.add_other and OTHER_TOKEN not in most:
            most.append(OTHER_TOKEN)

        self.items = most
        self.vocab = {k: i for i, k in enumerate(self.items)}

    def dim(self) -> int:
        return len(self.items)

    def one_hot(self, v: str) -> np.ndarray:
        x = np.zeros(self.dim(), dtype=np.float32)
        key = v if v in self.vocab else (OTHER_TOKEN if OTHER_TOKEN in self.vocab else MISSING_TOKEN)
        x[self.vocab.get(key, 0)] = 1.0
        return x


class TopKMultiHot:
    def __init__(self, k: int = 200):
        self.k = k
        self.items: List[str] = []
        self.vocab: Dict[str, int] = {}

    def fit_from_values(self, values: List[str]) -> None:
        cnt = Counter(values)
        most = [k for k, _ in cnt.most_common(self.k)]
        self.items = most
        self.vocab = {k: i for i, k in enumerate(self.items)}

    def dim(self) -> int:
        return len(self.items)

    def multi_hot(self, values: List[str]) -> np.ndarray:
        x = np.zeros(self.dim(), dtype=np.float32)
        for v in values:
            if v in self.vocab:
                x[self.vocab[v]] = 1.0
        return x


class TabularVectorizer:
    def __init__(
        self,
        window_hours: int = 48,
        use_topk_drug: bool = True,
        topk_drug: int = 200,
        max_vocab: int = 50,
    ):
        self.window_hours = window_hours
        self.use_topk_drug = use_topk_drug

        self.v_gender = CategoricalVocab(max_size=5, add_other=False)
        self.v_adm_type = CategoricalVocab(max_size=max_vocab)
        self.v_adm_loc = CategoricalVocab(max_size=max_vocab)
        self.v_insurance = CategoricalVocab(max_size=max_vocab)
        self.v_language = CategoricalVocab(max_size=max_vocab)
        self.v_marital = CategoricalVocab(max_size=max_vocab)
        self.v_careunit = CategoricalVocab(max_size=max_vocab)

        self.v_drug = TopKMultiHot(k=topk_drug)

        self.feature_names: List[str] = []
        self._dim: Optional[int] = None

    def fit(self, samples: List[Dict[str, Any]]) -> None:
        genders: List[str] = []
        adm_types: List[str] = []
        adm_locs: List[str] = []
        ins: List[str] = []
        langs: List[str] = []
        maritals: List[str] = []
        careunits: List[str] = []
        drug_values: List[str] = []

        for s in samples:
            t = _to_ts(s.get("cxr_time"))
            tab = s.get("tabular", {}) or {}

            df = tab.get("demographics")
            if df is not None and len(df) > 0 and "gender" in df.columns:
                genders.append(_safe_str(df.iloc[0]["gender"]))
            else:
                genders.append(MISSING_TOKEN)

            adf = tab.get("admissions")
            if adf is not None and len(adf) > 0:
                row = _select_latest_row_before(adf, "admittime", t)
                if row is not None:
                    adm_types.append(_safe_str(row.get("admission_type")))
                    adm_locs.append(_safe_str(row.get("admission_location")))
                    ins.append(_safe_str(row.get("insurance")))
                    langs.append(_safe_str(row.get("language")))
                    maritals.append(_safe_str(row.get("marital_status")))
                else:
                    adm_types.append(MISSING_TOKEN)
                    adm_locs.append(MISSING_TOKEN)
                    ins.append(MISSING_TOKEN)
                    langs.append(MISSING_TOKEN)
                    maritals.append(MISSING_TOKEN)
            else:
                adm_types.append(MISSING_TOKEN)
                adm_locs.append(MISSING_TOKEN)
                ins.append(MISSING_TOKEN)
                langs.append(MISSING_TOKEN)
                maritals.append(MISSING_TOKEN)

            tdf = tab.get("transfers")
            if tdf is not None and len(tdf) > 0:
                row = _select_active_interval_row(tdf, "intime", "outtime", t)
                careunits.append(_safe_str(row.get("careunit")) if row is not None else MISSING_TOKEN)
            else:
                careunits.append(MISSING_TOKEN)

            pdf = tab.get("prescriptions")
            if pdf is not None and len(pdf) > 0:
                w = _count_window_rows(pdf, "starttime", t, self.window_hours)
                if len(w) > 0 and "drug" in w.columns:
                    drug_values.extend([_safe_str(x) for x in w["drug"].tolist()])

        self.v_gender.fit_from_values(genders)
        self.v_adm_type.fit_from_values(adm_types)
        self.v_adm_loc.fit_from_values(adm_locs)
        self.v_insurance.fit_from_values(ins)
        self.v_language.fit_from_values(langs)
        self.v_marital.fit_from_values(maritals)
        self.v_careunit.fit_from_values(careunits)
        if self.use_topk_drug:
            self.v_drug.fit_from_values(drug_values)

        self.feature_names = []
        self.feature_names += ["age", "delta_admit_hours", "is_in_icu", "delta_icu_intime_hours"]
        self.feature_names += ["rx_count_48h", "rx_unique_drug_48h", "rx_unique_route_48h", "rx_unique_type_48h"]
        self.feature_names += ["proc_count_48h", "proc_unique_icd_48h"]

        self.feature_names += [f"gender={x}" for x in self.v_gender.items]
        self.feature_names += [f"adm_type={x}" for x in self.v_adm_type.items]
        self.feature_names += [f"adm_loc={x}" for x in self.v_adm_loc.items]
        self.feature_names += [f"insurance={x}" for x in self.v_insurance.items]
        self.feature_names += [f"language={x}" for x in self.v_language.items]
        self.feature_names += [f"marital={x}" for x in self.v_marital.items]
        self.feature_names += [f"careunit={x}" for x in self.v_careunit.items]

        if self.use_topk_drug:
            self.feature_names += [f"drug@48h={x}" for x in self.v_drug.items]

        self._dim = len(self.feature_names)

    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("TabularVectorizer not fitted. Call fit(samples) first.")
        return self._dim

    def transform_one(self, tabular: Dict[str, Any], cxr_time: Any) -> np.ndarray:
        t = _to_ts(cxr_time)
        tab = tabular or {}

        feats: List[np.ndarray] = []

        age = 0.0
        ddf = tab.get("demographics")
        if ddf is not None and len(ddf) > 0 and "anchor_age" in ddf.columns:
            try:
                age = float(ddf.iloc[0]["anchor_age"])
            except Exception:
                age = 0.0

        delta_admit_hours = 0.0
        adm_type = MISSING_TOKEN
        adm_loc = MISSING_TOKEN
        insurance = MISSING_TOKEN
        language = MISSING_TOKEN
        marital = MISSING_TOKEN

        adf = tab.get("admissions")
        if adf is not None and len(adf) > 0 and not pd.isna(t):
            row = _select_latest_row_before(adf, "admittime", t)
            if row is not None:
                at = _to_ts(row.get("admittime"))
                if not pd.isna(at):
                    delta_admit_hours = float((t - at) / pd.Timedelta(hours=1))
                adm_type = _safe_str(row.get("admission_type"))
                adm_loc = _safe_str(row.get("admission_location"))
                insurance = _safe_str(row.get("insurance"))
                language = _safe_str(row.get("language"))
                marital = _safe_str(row.get("marital_status"))

        is_in_icu = 0.0
        delta_icu_intime_hours = 0.0
        icu_careunit = MISSING_TOKEN

        icu = tab.get("icustays")
        if icu is not None and len(icu) > 0 and not pd.isna(t):
            row = _select_active_interval_row(icu, "intime", "outtime", t)
            if row is not None:
                is_in_icu = 1.0
                it = _to_ts(row.get("intime"))
                if not pd.isna(it):
                    delta_icu_intime_hours = float((t - it) / pd.Timedelta(hours=1))
                icu_careunit = _safe_str(row.get("first_careunit"))

        cur_careunit = MISSING_TOKEN
        tr = tab.get("transfers")
        if tr is not None and len(tr) > 0 and not pd.isna(t):
            row = _select_active_interval_row(tr, "intime", "outtime", t)
            if row is not None:
                cur_careunit = _safe_str(row.get("careunit"))

        rx_count = rx_uniq_drug = rx_uniq_route = rx_uniq_type = 0.0
        rx_drugs_for_topk: List[str] = []
        rx = tab.get("prescriptions")
        if rx is not None and len(rx) > 0 and not pd.isna(t):
            w = _count_window_rows(rx, "starttime", t, self.window_hours)
            rx_count = float(len(w))
            if len(w) > 0:
                if "drug" in w.columns:
                    vals = [_safe_str(x) for x in w["drug"].tolist()]
                    rx_drugs_for_topk = vals
                    rx_uniq_drug = float(len(set(vals)))
                if "route" in w.columns:
                    rx_uniq_route = float(len(set(_safe_str(x) for x in w["route"].tolist())))
                if "drug_type" in w.columns:
                    rx_uniq_type = float(len(set(_safe_str(x) for x in w["drug_type"].tolist())))

        proc_count = proc_uniq = 0.0
        proc = tab.get("procedures_icd")
        if proc is not None and len(proc) > 0 and not pd.isna(t):
            if "chartdate" in proc.columns:
                proc2 = proc.copy()
                proc2["chartdate"] = pd.to_datetime(proc2["chartdate"], errors="coerce")
                t0 = t - pd.Timedelta(hours=self.window_hours)
                m = (proc2["chartdate"] <= t) & (proc2["chartdate"] >= t0)
                w = proc2.loc[m]
            else:
                w = proc.iloc[0:0]

            proc_count = float(len(w))
            if len(w) > 0 and "icd_code" in w.columns:
                proc_uniq = float(len(set(_safe_str(x) for x in w["icd_code"].tolist())))

        feats.append(
            np.array(
                [
                    age,
                    delta_admit_hours,
                    is_in_icu,
                    delta_icu_intime_hours,
                    rx_count,
                    rx_uniq_drug,
                    rx_uniq_route,
                    rx_uniq_type,
                    proc_count,
                    proc_uniq,
                ],
                dtype=np.float32,
            )
        )

        gender = MISSING_TOKEN
        if ddf is not None and len(ddf) > 0 and "gender" in ddf.columns:
            gender = _safe_str(ddf.iloc[0]["gender"])
        feats.append(self.v_gender.one_hot(gender))

        feats.append(self.v_adm_type.one_hot(adm_type))
        feats.append(self.v_adm_loc.one_hot(adm_loc))
        feats.append(self.v_insurance.one_hot(insurance))
        feats.append(self.v_language.one_hot(language))
        feats.append(self.v_marital.one_hot(marital))

        careunit = cur_careunit if cur_careunit != MISSING_TOKEN else icu_careunit
        feats.append(self.v_careunit.one_hot(careunit))

        if self.use_topk_drug:
            feats.append(self.v_drug.multi_hot(rx_drugs_for_topk))

        x = np.concatenate(feats, axis=0).astype(np.float32)
        if self._dim is not None and x.shape[0] != self._dim:
            raise RuntimeError(f"Tab vector dim mismatch: got {x.shape[0]} expected {self._dim}")
        return x

    def transform_batch(self, batch: Dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
        tab_list = batch["tabular"]
        times = batch["cxr_time"]
        xs = [self.transform_one(tab_list[i], times[i]) for i in range(len(tab_list))]
        x = torch.from_numpy(np.stack(xs, axis=0))
        if device is not None:
            x = x.to(device)
        return x


def build_ts_schema_from_cfg_multi(
    TS_CFG: Dict[str, Dict[str, Any]],
    *,
    candidate_value_cols: List[str] = ("valuenum", "value", "amount", "rate", "patientweight"),
    global_value_allowlist: Optional[List[str]] = None,
    table_value_allowlist: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    global_allow: Optional[Set[str]] = set(global_value_allowlist) if global_value_allowlist else None
    per_table_allow: Dict[str, Set[str]] = {k: set(v) for k, v in (table_value_allowlist or {}).items()}

    schema: Dict[str, Dict[str, Any]] = {}

    for table, cfg in TS_CFG.items():
        usecols = cfg.get("usecols", []) or []

        if "itemid" not in usecols:
            continue
        id_col = "itemid"

        if "charttime" in usecols:
            time_col = "charttime"
            end_col = None
        elif "starttime" in usecols:
            time_col = "starttime"
            end_col = "endtime" if "endtime" in usecols else None
        else:
            continue

        found = [c for c in candidate_value_cols if c in usecols]
        if not found:
            continue

        if global_allow is not None:
            found = [c for c in found if c in global_allow]

        if table in per_table_allow:
            allow_set = per_table_allow[table]
            found = [c for c in found if c in allow_set]

        if not found:
            continue

        schema[table] = {
            "time_col": time_col,
            "end_col": end_col,
            "id_col": id_col,
            "value_cols": found,
        }

    return schema


class TimeseriesVectorizer:
    def __init__(
        self,
        ts_schema: Dict[str, Dict[str, Any]],
        window_hours: int = 48,
        topk_per_table: int = 256,
        stats: Tuple[str, ...] = ("count", "mean", "std", "min", "max", "last"),
        enabled_value_cols_global: Optional[List[str]] = None,
        enabled_value_cols_by_table: Optional[Dict[str, List[str]]] = None,
    ):
        self.ts_schema_raw = ts_schema
        self.window_hours = window_hours
        self.topk_per_table = topk_per_table
        self.stats = stats

        self.enabled_value_cols_global = set(enabled_value_cols_global) if enabled_value_cols_global else None
        self.enabled_value_cols_by_table = {k: set(v) for k, v in (enabled_value_cols_by_table or {}).items()}

        self.top_items: Dict[str, List[int]] = {}
        self.feature_names: List[str] = []
        self._dim: Optional[int] = None

        self.ts_schema = self._normalize_schema(self.ts_schema_raw)

    def _normalize_schema(self, schema: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for table, cfg in schema.items():
            vcols = list(cfg.get("value_cols", []) or [])
            if not vcols:
                continue

            if self.enabled_value_cols_global is not None:
                vcols = [c for c in vcols if c in self.enabled_value_cols_global]

            if table in self.enabled_value_cols_by_table:
                allow = self.enabled_value_cols_by_table[table]
                vcols = [c for c in vcols if c in allow]

            if not vcols:
                continue

            out[table] = {
                "time_col": cfg["time_col"],
                "end_col": cfg.get("end_col"),
                "id_col": cfg["id_col"],
                "value_cols": vcols,
            }
        return out

    def _filter_window(self, df: pd.DataFrame, t: pd.Timestamp, cfg: Dict[str, Any]) -> pd.DataFrame:
        if df is None or len(df) == 0 or pd.isna(t):
            return df.iloc[0:0] if df is not None else pd.DataFrame()

        t0 = t - pd.Timedelta(hours=self.window_hours)
        time_col = cfg["time_col"]
        end_col = cfg.get("end_col")

        if time_col not in df.columns:
            return df.iloc[0:0]

        if end_col is None:
            tt = pd.to_datetime(df[time_col], errors="coerce")
            m = (tt <= t) & (tt >= t0)
            out = df.loc[m].copy()
            out["_t"] = pd.to_datetime(out[time_col], errors="coerce")
            return out

        st = pd.to_datetime(df[time_col], errors="coerce")
        if end_col in df.columns:
            et = pd.to_datetime(df[end_col], errors="coerce")
            et_filled = et.fillna(st)
        else:
            et_filled = st

        m = (st <= t) & (et_filled >= t0)
        out = df.loc[m].copy()
        out["_t"] = st
        return out

    def fit(self, samples: List[Dict[str, Any]]) -> None:
        counters = {t: Counter() for t in self.ts_schema.keys()}

        for s in samples:
            t = _to_ts(s.get("cxr_time"))
            ts = s.get("timeseries", {}) or {}

            for table, cfg in self.ts_schema.items():
                df = ts.get(table)
                if df is None or len(df) == 0:
                    continue

                w = self._filter_window(df, t, cfg)
                if len(w) == 0:
                    continue

                id_col = cfg["id_col"]
                if id_col not in w.columns:
                    continue

                vals = pd.to_numeric(w[id_col], errors="coerce").dropna().astype("int64").tolist()
                counters[table].update(vals)

        self.top_items = {
            t: [k for k, _ in cnt.most_common(self.topk_per_table)]
            for t, cnt in counters.items()
        }

        self.feature_names = []
        for table, cfg in self.ts_schema.items():
            vcols = cfg["value_cols"]

            self.feature_names += [
                f"{table}__total_count",
                f"{table}__unique_itemid",
            ]
            for vc in vcols:
                self.feature_names.append(f"{table}__missing_value_ratio__{vc}")

            for itemid in self.top_items.get(table, []):
                for vc in vcols:
                    for st in self.stats:
                        self.feature_names.append(f"{table}__itemid={itemid}__{vc}__{st}")

        self._dim = len(self.feature_names)

    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("TimeseriesVectorizer not fitted.")
        return self._dim

    def transform_one(self, timeseries: Dict[str, Any], cxr_time: Any) -> np.ndarray:
        if self._dim is None:
            raise RuntimeError("TimeseriesVectorizer not fitted. Call fit() first.")

        t = _to_ts(cxr_time)
        ts = timeseries or {}

        feats: List[np.ndarray] = []

        for table, cfg in self.ts_schema.items():
            df = ts.get(table)
            if df is None:
                df = pd.DataFrame()

            w = self._filter_window(df, t, cfg)

            id_col = cfg["id_col"]
            vcols = cfg["value_cols"]

            if len(w) > 0 and id_col in w.columns:
                item = pd.to_numeric(w[id_col], errors="coerce")
            else:
                item = pd.Series([], dtype="float64")

            total_count = float(len(w))
            unique_itemid = float(item.dropna().nunique()) if len(item) > 0 else 0.0
            feats.append(np.array([total_count, unique_itemid], dtype=np.float32))

            for vc in vcols:
                if len(w) > 0 and vc in w.columns:
                    val = pd.to_numeric(w[vc], errors="coerce")
                    mr = float(val.isna().mean()) if len(val) > 0 else 1.0
                else:
                    mr = 1.0
                feats.append(np.array([mr], dtype=np.float32))

            top_items = self.top_items.get(table, [])
            if len(w) == 0 or len(top_items) == 0 or id_col not in w.columns:
                feats.append(np.zeros(len(top_items) * len(vcols) * len(self.stats), dtype=np.float32))
                continue

            w2 = w.copy()
            w2["_item"] = pd.to_numeric(w2[id_col], errors="coerce").astype("Int64")
            w2 = w2[w2["_item"].notna()].copy()
            if len(w2) == 0:
                feats.append(np.zeros(len(top_items) * len(vcols) * len(self.stats), dtype=np.float32))
                continue

            w2["_item"] = w2["_item"].astype("int64")
            w2 = w2.sort_values("_t", ascending=True)

            pos_item = {int(it): i for i, it in enumerate(top_items)}
            pos_keys = set(pos_item.keys())
            w2 = w2[w2["_item"].isin(pos_keys)]
            if len(w2) == 0:
                feats.append(np.zeros(len(top_items) * len(vcols) * len(self.stats), dtype=np.float32))
                continue

            for vc in vcols:
                if vc in w2.columns:
                    w2[f"__val__{vc}"] = pd.to_numeric(w2[vc], errors="coerce")
                else:
                    w2[f"__val__{vc}"] = np.nan

            out_block = np.zeros(len(top_items) * len(vcols) * len(self.stats), dtype=np.float32)

            g = w2.groupby("_item", sort=False)
            V = len(vcols)
            S = len(self.stats)

            for itemid, sub in g:
                ii = pos_item.get(int(itemid))
                if ii is None:
                    continue

                for vj, vc in enumerate(vcols):
                    vals = sub[f"__val__{vc}"].dropna()

                    stats_vals: List[float] = []
                    for st in self.stats:
                        if st == "count":
                            stats_vals.append(float(len(vals)))
                        elif st == "mean":
                            stats_vals.append(float(vals.mean()) if len(vals) > 0 else 0.0)
                        elif st == "std":
                            stats_vals.append(float(vals.std(ddof=0)) if len(vals) > 0 else 0.0)
                        elif st == "min":
                            stats_vals.append(float(vals.min()) if len(vals) > 0 else 0.0)
                        elif st == "max":
                            stats_vals.append(float(vals.max()) if len(vals) > 0 else 0.0)
                        elif st == "last":
                            stats_vals.append(float(vals.iloc[-1]) if len(vals) > 0 else 0.0)
                        else:
                            raise ValueError(f"unknown stat: {st}")

                    base = (ii * V + vj) * S
                    out_block[base : base + S] = np.array(stats_vals, dtype=np.float32)

            feats.append(out_block)

        x = np.concatenate(feats, axis=0).astype(np.float32)
        if x.shape[0] != self._dim:
            raise RuntimeError(f"TS vector dim mismatch: got {x.shape[0]} expected {self._dim}")
        return x

    def save_feature_names(self, path: str) -> None:
        if self._dim is None:
            raise RuntimeError("TimeseriesVectorizer not fitted.")
        with open(path, "w", encoding="utf-8") as f:
            for i, n in enumerate(self.feature_names):
                f.write(f"{i}\t{n}\n")

