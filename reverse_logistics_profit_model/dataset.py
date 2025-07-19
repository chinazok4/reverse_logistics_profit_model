from pathlib import Path
import ast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from loguru import logger
import typer


from reverse_logistics_profit_model.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

# ─── Helper Functions ──────────────────────────────────────────────────────────

def parse_list_literal(s: str) -> list[str]:
    """Safely turn a stringified Python list into a lowercase list of strings."""
    if isinstance(s, list): return [str(elem).strip().lower() for elem in s if isinstance(elem, str)]
    if not isinstance(s, str): return []
    try:
        items = ast.literal_eval(s)
    except Exception:
        return []
    return [str(item).strip().lower() for item in items if isinstance(item, str)]


def binarize_multilabel(col: pd.Series, prefix: str) -> pd.DataFrame:
    """One-hot encode a Series of lists via MultiLabelBinarizer."""
    mlb = MultiLabelBinarizer()
    arr = mlb.fit_transform(col)
    cols = [f"{prefix}_{c}" for c in mlb.classes_]
    return pd.DataFrame(arr, columns=cols, index=col.index)


def rare_lvl_grouping(col: pd.Series, cum_df: pd. DataFrame, cutoff_pct: float) -> None:
    cutoff = cum_df[cum_df['cumulative_pct'] >= cutoff_pct].index[0]
    top = cum_df.loc[:cutoff].index.tolist()
    col.where(col.isin(top), 'other', inplace=True)


def count_cumulative_calc(col: pd.Series) -> pd.DataFrame:
    counts = col.value_counts()
    cum_pct = counts.cumsum() / counts.sum()
    return pd.DataFrame({'count': counts, 'cumulative_pct': cum_pct.round(3)})


def preprocess_poshmark(path: Path) -> pd.DataFrame:
    """Load raw Poshmark CSV, clean and feature-engineer, return ready-to-model DataFrame."""
    df = pd.read_csv(path)

    # 1) Deduplicate exact URLs
    df.drop_duplicates(subset="url", inplace=True)

    # 2) Response label: sold within 90d → 1; unsold past 90d → 0
    df["listing_date"] = pd.to_datetime(df["listing_date"])
    ref_date = pd.Timestamp(2025,6,25)
    df["days_since_listing"] = (ref_date - df["listing_date"]).dt.days
    sold_mask  = (df["days_since_listing"] < 90) & df["sold"]
    unsold_mask= (df["days_since_listing"] > 90) & ~df["sold"]
    df = df[sold_mask | unsold_mask].copy()
    df["sellable"] = sold_mask.astype(int)

    # 3) Clean & cast listing_price to float
    df["listing_price"] = (
        pd.to_numeric(
            df["listing_price"].astype(str)
              .str.replace(r"[^\d]", "", regex=True),
            errors="coerce"
        )
        .astype(float)
    )

    # 4) Compute price_drop fraction (percent vs absolute)
    pct_mask = df["price_drop"].astype(str).str.contains("%", na=False)
    pct_vals = (
        df.loc[pct_mask, "price_drop"]
          .astype(str)
          .str.replace(",", "", regex=False)
          .str.rstrip("%")
          .astype(float)
          .div(100)
          .fillna(0)
    )
    abs_raw  = df.loc[~pct_mask, "price_drop"].astype(str).str.replace(",", "", regex=False)
    abs_vals = pd.to_numeric(abs_raw, errors="coerce")
    frac_abs = ((abs_vals - df.loc[~pct_mask, "listing_price"])
                 .div(abs_vals).fillna(0))
    df["price_drop"] = 0.0
    df.loc[pct_vals.index,   "price_drop"] = pct_vals
    df.loc[frac_abs.index,   "price_drop"] = frac_abs

    # 5) Split category into department & categories (lowercased)
    def split_cat(s):
        try:
            a = ast.literal_eval(s)
            return pd.Series({
                "department":   a[0].strip().lower(),
                "categories":   a[1].strip().lower()
            })
        except Exception:
            return pd.Series({"department":"unknown", "categories":"unknown"})
    df = df.join(df["category"].apply(split_cat))

    # 6) Filter out undesired categories
    df = df[~df["categories"].isin(
        {"makeup","skincare","bath & body","toys","other"}
    )].copy()

    # 7) Normalize & impute text fields
    text_cols = ["discounted_shipping","brand","department","categories", "sub_category"] 
    for c in text_cols:
        df[c] = (
            df[c].fillna("")
                 .astype(str)
                 .str.lower()
                 .str.strip()
                 .replace("", "unknown")
        )

    # 8) Rare-level grouping
    brand_df = count_cumulative_calc(df['brand'])
    rare_lvl_grouping(df['brand'], brand_df, 0.663)

    df['style_tags_list'] = df['style_tags'].apply(parse_list_literal)
    all_tags = [tag for sublist in df['style_tags_list'] for tag in sublist]
    tag_df = count_cumulative_calc(pd.Series(all_tags))
    rare_lvl_grouping(pd.Series(all_tags), tag_df, 0.741)

    subcat_df = count_cumulative_calc(df['sub_category'])
    rare_lvl_grouping(df['sub_category'], subcat_df, 0.982)
 

    # 9) One-hot single-label categoricals
    df = pd.get_dummies(
        df,
        columns=["discounted_shipping","brand","department","categories", "sub_category"],
        prefix=["ship","brand","dept","cat", "subcat"]
    )

    # Multi-label encode style_tags
    df = pd.concat([df, binarize_multilabel(df["style_tags_list"], "tag")], axis=1)

    df['color_list'] = df['color'].apply(parse_list_literal)
    df = pd.concat([df, binarize_multilabel(df['color_list'],'color')], axis=1)

    # Outlier treatment
    df["log_price"] = np.log1p(df["listing_price"])
    upper_pd = df["price_drop"].quantile(0.99)
    df["price_drop"] = df["price_drop"].clip(upper=upper_pd)

    # Scale numerics
    scaler = StandardScaler()
    num_cols = ["log_price","price_drop"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Drop intermediates
    to_drop = [
        "url","category", "listing_date", "listing_price", "color", "sold","size",
        "style_tags","style_tags_list", "color_list", "days_since_listing"
    ]
    df.drop(columns=[c for c in to_drop if c in df], inplace=True)

    return df

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "poshmark_sample.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    processed = preprocess_poshmark(input_path)
    processed.to_csv(output_path, index=False)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
