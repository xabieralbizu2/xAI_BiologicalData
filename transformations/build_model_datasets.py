"""
Build the modelling datasets from the Yeo-Johnson-normalized data.

Replaces the manual steps that used to sit between preprocessing and the models
(drop_boolean_columns.ipynb plus renaming/adding columns by hand). From
normalization/dataset_normalized.csv it writes, into transformations/src/:

  dataset_normalized_eda.csv  normalized data with the status column renamed to
                              'Extant/Extinct ' (the input of eda/variable_analysis.py)
  dataset_nbool_std.csv       continuous variables only (one-hot columns dropped)
                              + Target_Specie, 'Extant/Extinct ', Specimen, ID
  dataset_con_grupos.csv      dataset_nbool_std + Target_group (genus)

Note: despite its historical name, dataset_nbool_std.csv is NOT standardized.
Every model standardizes inside its own cross-validation folds, so no scaling
statistic leaks from the held-out specimen.

Usage:  python transformations/build_model_datasets.py [--input PATH] [--outdir DIR]
"""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_EXCEL = HERE / "one_hot" / "PALAEO3_microtexture2.xlsx"
RAW_SHEET, RAW_SKIPROWS = "3ºOPTION", 3      # same sheet and header row as one_hot.py

STATUS_COL = "Extant/Extinct "               # trailing space kept on purpose: the
                                             # models and EDA scripts expect it
META_ORDER = ["Target_Specie", STATUS_COL, "Specimen", "ID"]


def specimen_ids(n_rows):
    """IDs from the raw Excel, in row order (for normalized files without an ID column)."""
    raw = pd.read_excel(RAW_EXCEL, sheet_name=RAW_SHEET, skiprows=RAW_SKIPROWS)
    raw = raw.dropna(how="all")
    if len(raw) != n_rows:
        raise SystemExit(f"Row mismatch: raw Excel has {len(raw)} rows, normalized data {n_rows}.")
    return raw["ID"].astype(str).values      # kept verbatim (a few IDs carry a leading space)


def build(norm: pd.DataFrame):
    norm = norm.copy()
    if "Status" in norm.columns:
        norm = norm.rename(columns={"Status": STATUS_COL})
    if STATUS_COL not in norm.columns or "Target_Specie" not in norm.columns:
        raise SystemExit("Input must contain 'Target_Specie' and 'Status' (or 'Extant/Extinct ').")

    if "ID" not in norm.columns:
        norm["ID"] = specimen_ids(len(norm))
    if "Specimen" not in norm.columns:
        norm["Specimen"] = norm["ID"]

    eda = norm.drop(columns=["ID", "Specimen"])

    # One-hot columns are stored as True/False: drop them, keep the continuous variables.
    continuous = [c for c in norm.select_dtypes(include="number").columns if c not in META_ORDER]
    nbool = pd.concat([norm[continuous], norm[META_ORDER]], axis=1)

    grupos = nbool.copy()
    grupos["Target_group"] = grupos["Target_Specie"].astype(str).str.split().str[0]
    return eda, nbool, grupos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=HERE / "normalization" / "dataset_normalized.csv")
    ap.add_argument("--outdir", type=Path, default=HERE / "src")
    args = ap.parse_args()

    norm = pd.read_csv(args.input)
    eda, nbool, grupos = build(norm)

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, df in [("dataset_normalized_eda.csv", eda),
                     ("dataset_nbool_std.csv", nbool),
                     ("dataset_con_grupos.csv", grupos)]:
        df.to_csv(args.outdir / name, index=False)
        print(f"  -> {args.outdir / name}  {df.shape}")
    n_cont = nbool.shape[1] - len(META_ORDER)
    print(f"Continuous variables: {n_cont} | specimens: {len(nbool)} "
          f"({(nbool[STATUS_COL].str.strip() == 'Extant').sum()} extant, "
          f"{(nbool[STATUS_COL].str.strip() == 'Extinct').sum()} fossil)")


if __name__ == "__main__":
    main()
