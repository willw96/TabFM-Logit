"""Split statistics quoted in Section 4: how many row-level test rows come
from people who also appear in training, and the sizes of the grouped splits.

    python paper/data_stats.py
"""
import numpy as np
import pandas as pd

from common import CACHED_DIR
from tabfm_logit import data as D


def share_in_train(train_ids, test_ids):
    seen = set(train_ids)
    return 100 * np.mean([i in seen for i in test_ids])


if __name__ == "__main__":
    # Row-level splits rebuilt with the identifiers carried along (the
    # identifiers do not affect the split).
    df = D._swissmetro_raw()
    df = df[(df["PURPOSE"] != 0) & (df["CHOICE"] != 0)].copy()
    rows = df[D.SM_FEATURES + ["CHOICE", "ID"]].dropna().assign(label=lambda d: d["CHOICE"].astype(int) - 1)
    tr, _, te = D.stratified_split(rows)
    print(f"Swissmetro row-level split: {share_in_train(tr['ID'], te['ID']):.1f}% of test rows come from "
          f"respondents who also appear in training")

    df = D._lpmc_raw()
    rows = df[D.LPMC_FEATURES + ["travel_mode", "household_id", "person_n"]].dropna()
    rows = rows.assign(label=lambda d: d["travel_mode"].astype(int) - 1)
    tr, _, te = D.stratified_split(rows)
    person = lambda d: list(zip(d["household_id"], d["person_n"]))  # noqa: E731
    print(f"LPMC row-level split: {share_in_train(person(tr), person(te)):.1f}% of test trips come from persons "
          f"in training, {share_in_train(tr['household_id'], te['household_id']):.1f}% from households in training")

    for ds, col in (("swissmetro_resp", "subject"), ("lpmc_hh", "household"), ("iot_wearables", "subject")):
        parts = [pd.read_parquet(CACHED_DIR / f"{ds}_{s}.parquet") for s in ("train", "val", "test")]
        print(f"{ds}: groups " + "/".join(str(p[col].nunique()) for p in parts)
              + ", rows " + "/".join(str(len(p)) for p in parts))
