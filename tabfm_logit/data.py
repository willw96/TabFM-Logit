"""
Datasets and train/validation/test splits.

Raw files go in ``data/raw/`` (see the README for sources):

    swissmetro.dat                      Swissmetro stated-preference survey
    lpmc.dat                            London Passenger Mode Choice (LPMC)
    IoT_DCE_Wearables-PLOS_ONE.dta      IoT-Wearables discrete-choice experiment

Splits are written to ``data/cached/{dataset}_{train,val,test}.parquet``.

Datasets (70/15/15 everywhere):

    swissmetro       row-level split, stratified by the chosen alternative
    lpmc             row-level split, stratified by the chosen alternative
    iot_wearables    subject-level split (each subject's rows go to one split)
    swissmetro_resp  respondent-level split of the Swissmetro rows
    lpmc_hh          household-level split of the LPMC rows

Each row is one choice task in wide format.

Usage:
    python -m tabfm_logit.data --dataset all
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CACHED_DIR = ROOT / "data" / "cached"

SEED = 42

# SHA-256 of the raw files the paper's splits were built from. A mismatch is
# reported as a warning: a different release of a file can change the splits.
RAW_SHA256 = {
    "swissmetro.dat": "27432693cf052985d79a950b4b888be3efca798fc89b0d3ffefe40608ede00f2",
    "lpmc.dat": "033b9eda6742ffa40716bcc4c32bf05cf3a9787058fb8335b65468ac48455338",
    "IoT_DCE_Wearables-PLOS_ONE.dta": "a74acddcef2b88cb58187e9e7c18e545f2a531c5879dfc8952e93a669cc0d381",
}


def _raw(name):
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See the README for where to download it.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != RAW_SHA256[name]:
        print(f"[warning] {name}: SHA-256 differs from the file used in the paper; splits may differ.")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────

def stratified_split(df, label_col="label", seed=SEED):
    """Row-level 70/15/15 split, stratified by the chosen alternative."""
    train_val, test = train_test_split(df, test_size=0.15, random_state=seed, stratify=df[label_col])
    train, val = train_test_split(train_val, test_size=0.15 / 0.85, random_state=seed,
                                  stratify=train_val[label_col])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def subject_level_split(df, subject_col, sizes=(0.70, 0.15, 0.15), seed=SEED):
    """70/15/15 split of subjects: each subject's rows go entirely to one split."""
    train_frac, val_frac, _ = sizes
    subjects = np.array(sorted(df[subject_col].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])
    train = df[df[subject_col].isin(train_subj)].reset_index(drop=True)
    val = df[df[subject_col].isin(val_subj)].reset_index(drop=True)
    test = df[df[subject_col].isin(test_subj)].reset_index(drop=True)
    return train, val, test


def _group_split(rows, group_col, seed=SEED):
    """70/15/15 split of groups (respondents or households); row order is kept."""
    groups = np.sort(rows[group_col].unique())
    rng = np.random.default_rng(seed)
    groups = groups[rng.permutation(len(groups))]
    n_tr = int(round(0.70 * len(groups)))
    n_va = int(round(0.15 * len(groups)))
    assign = {"train": set(groups[:n_tr]), "val": set(groups[n_tr:n_tr + n_va]),
              "test": set(groups[n_tr + n_va:])}
    return tuple(rows[rows[group_col].isin(assign[s])].reset_index(drop=True)
                 for s in ("train", "val", "test"))


def _attach_ids(rows, raw, key_fn_rows, key_fn_raw, id_fn):
    """Attach identifiers from the raw file to the rows of the row-level splits.

    Rows are matched to raw records by their feature values and choice. Records
    with identical values are matched in file order; they are identical rows,
    so the matching does not change any split's contents."""
    queue = {}
    for k, ids in zip(key_fn_raw(raw), id_fn(raw)):
        queue.setdefault(k, []).append(ids)
    out = []
    for k in key_fn_rows(rows):
        if not queue.get(k):
            raise ValueError("a row has no matching raw record")
        out.append(queue[k].pop(0))
    return out


def get_Xy(df, features, label_col="label"):
    """Extract X (n_obs, n_features) and y (n_obs,) from a wide-format DataFrame."""
    X = df[features].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Swissmetro: 3 alternatives (train, Swissmetro, car)
# ──────────────────────────────────────────────────────────────────────────────

SM_FEATURES = [
    "TRAIN_TT", "TRAIN_CO", "TRAIN_HE",
    "SM_TT", "SM_CO", "SM_HE",
    "CAR_TT", "CAR_CO",
    "TRAIN_AV", "SM_AV", "CAR_AV",
    "AGE", "INCOME", "GA", "LUGGAGE", "PURPOSE",
]
SM_MODES = {0: "train", 1: "sm", 2: "car"}
_SM_SCALED = ["TRAIN_TT", "TRAIN_CO", "TRAIN_HE", "SM_TT", "SM_CO", "SM_HE", "CAR_TT", "CAR_CO"]
_SM_KEY = _SM_SCALED + ["TRAIN_AV", "SM_AV", "CAR_AV", "AGE", "INCOME", "GA", "LUGGAGE", "PURPOSE", "CHOICE"]


def _swissmetro_raw():
    return pd.read_csv(_raw("swissmetro.dat"), sep="\t")


def _swissmetro_rows():
    """Swissmetro rows with a known purpose and choice. Times, costs and
    headways are divided by 100."""
    df = _swissmetro_raw()
    df = df[df["PURPOSE"] != 0]
    df = df[df["CHOICE"] != 0]
    for col in _SM_SCALED:
        df[col] = df[col] / 100.0
    out = df[SM_FEATURES + ["CHOICE"]].copy().dropna()
    out["label"] = out["CHOICE"].astype(int) - 1
    return out


def load_swissmetro():
    return stratified_split(_swissmetro_rows())


def _sm_key(df, from_raw):
    d = df[_SM_KEY].copy()
    for c in _SM_SCALED:
        d[c] = (d[c] / 100.0 if from_raw else d[c]).round(2)
    return d.astype(str).agg("|".join, axis=1)


def load_swissmetro_resp(seed=SEED):
    """Swissmetro split by respondent: the rows of the row-level splits, with
    each respondent's rows in exactly one split. Adds a ``subject`` column."""
    rows = pd.concat(load_swissmetro(), ignore_index=True)
    raw = _swissmetro_raw()
    raw = raw[(raw["PURPOSE"] != 0) & (raw["CHOICE"] != 0)].reset_index(drop=True)
    ids = _attach_ids(rows, raw, lambda d: _sm_key(d, False), lambda d: _sm_key(d, True),
                      lambda d: d["ID"].astype(int))
    rows["subject"] = np.asarray(ids, dtype=np.int64)
    return _group_split(rows, "subject", seed)


# ──────────────────────────────────────────────────────────────────────────────
# LPMC: 4 alternatives (walk, cycle, public transport, drive)
# ──────────────────────────────────────────────────────────────────────────────

LPMC_FEATURES = [
    "dur_walking", "dur_cycling", "dur_pt_total",
    "dur_driving", "cost_transit", "cost_driving",
    "age", "female", "driving_license", "car_ownership", "distance",
]
LPMC_MODES = {0: "walk", 1: "cycle", 2: "pt", 3: "drive"}


def _lpmc_raw():
    df = pd.read_csv(_raw("lpmc.dat"), sep="\t")
    df["dur_pt_total"] = df[["dur_pt_access", "dur_pt_rail", "dur_pt_bus", "dur_pt_int"]].sum(axis=1)
    df["cost_driving"] = df[["cost_driving_fuel", "cost_driving_ccharge"]].sum(axis=1)
    return df


def load_lpmc():
    df = _lpmc_raw()
    out = df[LPMC_FEATURES + ["travel_mode"]].copy().dropna()
    out["label"] = out["travel_mode"].astype(int) - 1
    return stratified_split(out)


_LPMC_KEY = ["dur_walking", "dur_cycling", "dur_driving", "cost_transit", "age", "female",
             "driving_license", "car_ownership", "distance", "travel_mode", "dur_pt_total", "cost_driving"]


def _lpmc_key(df):
    k = df[_LPMC_KEY].copy()
    for c in _LPMC_KEY:
        if k[c].dtype.kind == "f":
            k[c] = k[c].round(5)
    return k.astype(str).agg("|".join, axis=1)


def load_lpmc_hh(seed=SEED):
    """LPMC split by household: the rows of the row-level splits, with each
    household's trips in exactly one split. Adds ``household`` and ``person``."""
    rows = pd.concat(load_lpmc(), ignore_index=True)
    raw = _lpmc_raw()
    ids = _attach_ids(rows, raw, _lpmc_key, _lpmc_key,
                      lambda d: list(zip(d["household_id"].astype(int), d["person_n"].astype(int))))
    rows["household"] = np.asarray([h for h, _ in ids], dtype=np.int64)
    rows["person"] = np.asarray([h * 100 + p for h, p in ids], dtype=np.int64)
    return _group_split(rows, "household", seed)


# ──────────────────────────────────────────────────────────────────────────────
# IoT-Wearables: 3 hypothetical wearable devices (Johnson et al., 2020).
# Long-format Stata file (3 rows per task), restricted to wearables, pivoted
# to wide format; the 7-way information condition becomes 6 dummies.
# ──────────────────────────────────────────────────────────────────────────────

IOT_PER_ALT_FEATURES = [
    "price1", "price2", "price3",
    "function1", "function2", "function3",
    "label1", "label2", "label3",
]
IOT_CONDITIONS = [
    "info_label_high_sec",  # reference category
    "SbD", "info_label_high", "Graded_D", "Graded_A", "Graded_G", "info_label_low",
]
IOT_CONDITION_DUMMIES = [f"cond_{c}" for c in IOT_CONDITIONS[1:]]
IOT_SUBJECT_FEATURES = ["age", "education", "Female", "secbehav"]
IOT_WEARABLES_FEATURES = IOT_PER_ALT_FEATURES + IOT_SUBJECT_FEATURES + IOT_CONDITION_DUMMIES
IOT_WEARABLES_MODES = {0: "alt1", 1: "alt2", 2: "alt3"}


def load_iot_wearables():
    df = pd.read_stata(_raw("IoT_DCE_Wearables-PLOS_ONE.dta"))
    df = df[df["stproducttested"] == "Wearable"].copy()
    sizes = df.groupby("choicetask").size()
    if not (sizes == 3).all():
        raise RuntimeError("expected exactly 3 alternatives per choice task")

    pivot = df.pivot_table(index="choicetask", columns="alternative",
                           values=["price", "function", "label"], aggfunc="first")
    pivot.columns = [f"{var}{int(alt)}" for var, alt in pivot.columns]
    pivot = pivot.reset_index()

    chosen = (df.loc[df["choice"] == 1, ["choicetask", "alternative"]]
              .rename(columns={"alternative": "label"}).copy())
    chosen["label"] = chosen["label"].astype(int) - 1
    pivot = pivot.merge(chosen, on="choicetask", how="left")
    if pivot["label"].isna().any():
        raise RuntimeError("some choice tasks have no chosen alternative")

    subj = df.drop_duplicates("choicetask")[
        ["choicetask", "subject", "condition", "age", "education", "Female", "secbehav"]]
    pivot = pivot.merge(subj, on="choicetask", how="left")
    for c in IOT_CONDITIONS[1:]:
        pivot[f"cond_{c}"] = (pivot["condition"] == c).astype(np.float32)

    out = pivot[["choicetask", "subject", "label"] + IOT_PER_ALT_FEATURES
                + IOT_SUBJECT_FEATURES + IOT_CONDITION_DUMMIES].copy()
    for col in IOT_PER_ALT_FEATURES + IOT_SUBJECT_FEATURES:
        out[col] = out[col].astype(np.float32)
    out["label"] = out["label"].astype(np.int64)
    return subject_level_split(out, "subject")


# ──────────────────────────────────────────────────────────────────────────────
# Registry and cached split I/O
# ──────────────────────────────────────────────────────────────────────────────

# Group column used for grouped cross-fitting and group-clustered intervals.
GROUP_COL = {
    "swissmetro_resp": "subject",
    "lpmc_hh": "household",
}

LOADERS = {
    "swissmetro": load_swissmetro,
    "lpmc": load_lpmc,
    "iot_wearables": load_iot_wearables,
    "swissmetro_resp": load_swissmetro_resp,
    "lpmc_hh": load_lpmc_hh,
}

DATASET_FEATURES = {
    "swissmetro": SM_FEATURES,
    "swissmetro_resp": SM_FEATURES,
    "lpmc": LPMC_FEATURES,
    "lpmc_hh": LPMC_FEATURES,
    "iot_wearables": IOT_WEARABLES_FEATURES,
}

DATASET_N_ALTS = {
    "swissmetro": 3, "swissmetro_resp": 3,
    "lpmc": 4, "lpmc_hh": 4,
    "iot_wearables": 3,
}


def splits_exist(dataset_name):
    return all((CACHED_DIR / f"{dataset_name}_{s}.parquet").exists() for s in ("train", "val", "test"))


def save_splits(dataset_name, force=False):
    if dataset_name not in LOADERS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Known: {list(LOADERS)}")
    if splits_exist(dataset_name) and not force:
        print(f"[{dataset_name}] cached splits exist (use --force to rebuild)")
        return
    CACHED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in zip(("train", "val", "test"), LOADERS[dataset_name]()):
        path = CACHED_DIR / f"{dataset_name}_{name}.parquet"
        split.to_parquet(path)
        print(f"  saved {path}  ({len(split)} rows)")


def load_splits(dataset_name):
    """Cached (train, val, test) splits; builds them on first use."""
    if not splits_exist(dataset_name):
        save_splits(dataset_name)
    return tuple(pd.read_parquet(CACHED_DIR / f"{dataset_name}_{s}.parquet") for s in ("train", "val", "test"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the train/validation/test splits.")
    p.add_argument("--dataset", choices=list(LOADERS) + ["all"], default="all")
    p.add_argument("--force", action="store_true", help="rebuild cached splits")
    args = p.parse_args()
    for name in (list(LOADERS) if args.dataset == "all" else [args.dataset]):
        save_splits(name, force=args.force)
        tr, va, te = load_splits(name)
        print(f"  {name}: train {len(tr)}  val {len(va)}  test {len(te)}")
