"""
Layer 2 entry point: are the fossils comparable with the extant specimens?

Runs the extant–fossil comparability analysis of variable_analysis.py on one
dataset and writes its outputs to --outdir:

  verosimilitud_gmm_1.csv          GMM fit on extant, log-likelihood of both populations
  verosimilitud_gmm_2.csv          GMM fit on fossils, log-likelihood of both populations
  distancias_emd.csv               per-variable Earth Mover's Distance, extant vs fossil
  significancia_pvalor.csv         per-variable test (t-test or Mann–Whitney)
  importancia_variables_pca.csv    PCA structure, extant vs fossil
  varianza_acumulada_pca_grupos.png

One-hot indicator columns are excluded: the analysis uses the 107 continuous
microtexture variables, as in the thesis. variable_analysis.py prints errors
instead of raising them, so this script checks that every output was written
and exits with an error otherwise.

Usage (from the eda/ folder):
  python run_layer2.py --input src/dataset_normalized.csv --outdir layer2/norm
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import variable_analysis as va  # noqa: E402

EXPECTED = ["verosimilitud_gmm_1.csv", "verosimilitud_gmm_2.csv", "distancias_emd.csv",
            "significancia_pvalor.csv", "importancia_variables_pca.csv",
            "varianza_acumulada_pca_grupos.png"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=HERE / "src" / "dataset_normalized.csv")
    ap.add_argument("--outdir", type=Path, default=HERE / "layer2")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if va.COLUMNA_ESTADO not in df.columns and "Status" in df.columns:
        df = df.rename(columns={"Status": va.COLUMNA_ESTADO})
    if va.COLUMNA_ESTADO not in df.columns:
        sys.exit(f"{args.input}: no '{va.COLUMNA_ESTADO.strip()}' or 'Status' column.")
    df = df.drop(columns=df.select_dtypes(include="bool").columns)
    n_vars = df.select_dtypes(include="number").shape[1]
    print(f"Layer 2 on {args.input.name}: {len(df)} specimens, {n_vars} continuous variables")

    args.outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        clean_csv = Path(tmp) / "layer2_input.csv"
        df.to_csv(clean_csv, index=False)
        va.ARCHIVO_CSV = str(clean_csv)
        cwd = os.getcwd()
        os.chdir(args.outdir)                # variable_analysis writes to the working directory
        try:
            va.generar_reportes_modularizados()
            va.plotear_varianza_acumulada_grupos()
        finally:
            os.chdir(cwd)

    missing = [f for f in EXPECTED if not (args.outdir / f).exists()]
    if missing:
        sys.exit(f"Layer 2 failed: missing outputs {missing} (see the error printed above).")

    # Summary: how well the extant-trained density explains each population.
    from scipy.stats import mannwhitneyu
    ll = pd.read_csv(args.outdir / "verosimilitud_gmm_1.csv")
    extant = ll.loc[ll["Estado"].str.strip() == "Extant", "Log_Likelihood"]
    fossil = ll.loc[ll["Estado"].str.strip() != "Extant", "Log_Likelihood"]
    p = mannwhitneyu(extant, fossil, alternative="two-sided").pvalue
    print(f"GMM fit on extant -> median log-likelihood: extant {extant.median():.3g}, "
          f"fossil {fossil.median():.3g} | Mann–Whitney p = {p:.2e}")
    print(f"Layer 2 outputs written to {args.outdir}")


if __name__ == "__main__":
    main()
