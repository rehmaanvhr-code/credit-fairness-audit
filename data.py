import os
import urllib.request

import pandas as pd

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv"
RAW = "data/german.csv"

COLS = ["checking", "duration", "history", "purpose", "amount", "savings", "employment",
        "installment_rate", "personal_status", "other_debtors", "residence_since",
        "property", "age", "other_plans", "housing", "n_credits", "job", "n_dependents",
        "telephone", "foreign_worker", "label"]


def load():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(RAW):
        urllib.request.urlretrieve(URL, RAW)
    df = pd.read_csv(RAW, header=None, names=COLS)
    # label is 1 good, 2 bad in the original file. 1 means approved here.
    df["label"] = (df["label"] == 1).astype(int)
    # attribute 9 codes marital status and sex together
    df["sex"] = df["personal_status"].map(
        {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"})
    # the age cut used in the fairness literature for this dataset
    df["age_group"] = (df["age"] > 25).map({True: "over 25", False: "25 or under"})
    return df


def features(df, drop_sensitive):
    X = df.drop(columns=["label", "sex", "age_group"])
    if drop_sensitive:
        X = X.drop(columns=["personal_status", "age"])
    X = pd.get_dummies(X, drop_first=True).astype(float)
    return X
