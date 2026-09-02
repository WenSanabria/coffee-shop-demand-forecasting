from pathlib import Path
from src.forecasting import run_pipeline

DATA = Path(__file__).parent / "data" / "sample_pos_transactions.csv"
models, results, train, test, cutoff, clean, model_df = run_pipeline(DATA)
print(f"Raw/clean records: {len(clean):,}")
print(f"Model rows: {len(model_df):,}")
print(f"Time split cutoff: {cutoff.date()}")
print("\nModel comparison:\n")
print(results.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
