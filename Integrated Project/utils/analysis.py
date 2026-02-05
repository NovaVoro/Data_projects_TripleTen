import pandas as pd

def summarize_stage_concentrations(df):
    rows = []
    stages = ["rougher", "primary_cleaner", "secondary_cleaner", "final"]
    metals = ["au", "ag", "pb"]

    for stage in stages:
        for metal in metals:
            cols = [c for c in df.columns if stage in c and metal in c]
            if cols:
                rows.append({
                    "stage": stage,
                    "metal": metal.upper(),
                    "median": df[cols].median(numeric_only=True).mean()
                })

    return pd.DataFrame(rows)