#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def compute_stats(df):
    """Add delta column and compute basic statistics."""
    df["delta"] = df["actual_time"] - df["scheduled_time"]

    stats = {
        "count": df["delta"].count(),
        "mean": df["delta"].mean(),
        "median": df["delta"].median(),
        "min": df["delta"].min(),
        "max": df["delta"].max(),
        "std_dev": df["delta"].std(ddof=1),  # sample std dev
    }
    return stats


def plot_histogram(df, out_path: Path):
    """Plot histogram of actual - scheduled and save to file."""
    plt.figure()
    plt.hist(df["delta"], bins=15)
    plt.xlabel("Delta (actual_time - scheduled_time) [seconds]")
    plt.ylabel("Frequency")
    plt.title("Histogram of Timing Differences")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze timing differences between scheduled and actual events."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to CSV file with columns: event_id,event_type,is_automatic,scheduled_time,actual_time",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=Path("delta_histogram.png"),
        help="Output path for histogram figure (default: delta_histogram.png)",
    )
    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.csv_path)

    # Compute stats
    stats = compute_stats(df)

    # Print stats
    print("Timing difference statistics (actual_time - scheduled_time):")
    for k, v in stats.items():
        print(f"  {k:>7}: {v:.9f}" if isinstance(v, float) else f"  {k:>7}: {v}")

    # Plot histogram
    plot_histogram(df, args.out_fig)
    print(f"\nHistogram saved to: {args.out_fig}")


if __name__ == "__main__":
    main()