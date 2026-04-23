"""
locust/traffic_profile.py
───────────────────────────────────────────────────────────────
Converts the pre-processed WorldCup98 invocation_count.csv into the
(second_offset, rps) profile that Locust and train_brain.py consume.

invocation_count.csv format (from nimamahmoudi/worldcup98-dataset):
  - index column : datetime string  e.g. "1998-04-30 00:01:00"
  - 'count'      : total requests in that 1-minute bucket
  - 'period'     : grouping label (not needed here)

Train/test split:
  --split train  →  April 30 - June 30 1998  (first ~61 days)
  --split test   →  July 1  - July 26 1998   (final ~26 days)
  --split all    →  entire file (default)

Usage
─────
  python traffic_profile.py \
      --csv   ../ai-controller/data/worldcup/invocation_count.csv \
      --out   traffic_profile_train.csv \
      --split train

  python traffic_profile.py \
      --csv   ../ai-controller/data/worldcup/invocation_count.csv \
      --out   traffic_profile_test.csv \
      --split test
"""

import argparse
import csv
import os

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
#  SPLIT BOUNDARIES
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_RANGES = {
    'train': ('1998-04-30', '1998-06-30 23:59:59'),
    'test':  ('1998-07-01', '1998-07-26 23:59:59'),
    'all':   (None, None),
}


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD + CONVERT
# ─────────────────────────────────────────────────────────────────────────────

def load_invocation_csv(csv_path: str, split: str = 'all',
                        max_rps: float | None = None) -> pd.DataFrame:
    """
    Load invocation_count.csv and return a DataFrame with columns:
        second_offset  (int)   — seconds since first row in this split
        rps            (float) — requests per second for that bucket
    """
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]

    if 'count' not in df.columns:
        raise ValueError(
            f"Expected a 'count' column in {csv_path}. "
            f"Found: {list(df.columns)}"
        )

    # Group by the datetime index in case there are duplicate minute entries
    df = df.groupby(df.index)['count'].sum().to_frame()
    df = df.sort_index()

    # Apply split filter
    start, end = SPLIT_RANGES.get(split, (None, None))
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]

    if df.empty:
        raise ValueError(
            f"No data after applying split='{split}'. "
            f"Check --split value and CSV date range."
        )

    # Each bucket is 1 minute wide → RPS = count / 60
    df['rps'] = df['count'] / 60.0

    # Convert datetime index to integer second offsets from the first row
    t0 = df.index[0]
    df['second_offset'] = ((df.index - t0).total_seconds()).astype(int)
    if max_rps is not None:
        original_peak = df['rps'].max()
        scale_factor  = max_rps / original_peak
        df['rps']     = (df['rps'] * scale_factor).round(2)
        print(f"Scaled RPS by {scale_factor:.5f}  "
            f"(original peak={original_peak:.1f} → target peak={max_rps:.1f})")

    print(f"Split '{split}': {len(df)} minute-buckets loaded  "
          f"({df.index[0]}  →  {df.index[-1]})")
    print(f"  Peak RPS : {df['rps'].max():.2f}")
    print(f"  Mean RPS : {df['rps'].mean():.2f}")
    print(f"  Min  RPS : {df['rps'].min():.2f}")
    

    return df[['second_offset', 'rps']].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

def save_profile(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Profile saved  →  {out_path}  ({len(df)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert invocation_count.csv → Locust traffic profile")
    parser.add_argument(
        '--csv', required=True,
        help='Path to invocation_count.csv from nimamahmoudi/worldcup98-dataset')
    parser.add_argument(
        '--out', default='locust/traffic_profile_train.csv',
        help='Output CSV path  (default: locust/traffic_profile_train.csv)')
    parser.add_argument(
        '--split', choices=['train', 'test', 'all'], default='all',
        help='train = Apr-Jun, test = Jul, all = entire file  (default: all)')
    parser.add_argument(
        '--max-rps', type=float, default=None,
        help='Normalise profile so peak RPS equals this value. '
            'Use 400 to match a 6-container testbed at 80%% load.'
    )
    args = parser.parse_args()

    df = load_invocation_csv(args.csv, split=args.split, max_rps=args.max_rps)
    save_profile(df, args.out)


if __name__ == '__main__':
    main()