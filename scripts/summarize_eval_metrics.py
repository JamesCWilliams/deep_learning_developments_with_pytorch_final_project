#!/usr/bin/env python
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_group_metrics(df: pd.DataFrame):
    """
    df: all episodes for a single (project, algo, env_id, ...) group.
    Returns a dict with intra/inter metrics and group size.

    Intra-agent:
      - per-run reward variance over episodes, then mean±std across runs
      - per-run mean duration_rate, then mean±std across runs

    Inter-agent:
      - variance/std of run-level mean returns
      - variance/std of run-level mean duration rates
    """
    # ensure duration_rate exists
    if "duration_rate" not in df.columns:
        if "max_episode_steps" in df.columns:
            df = df.copy()
            df["duration_rate"] = df["episode_length"] / df["max_episode_steps"]
        else:
            raise ValueError(
                "duration_rate not in CSV and max_episode_steps missing; "
                "rerun evaluation with duration_rate logging."
            )

    run_groups = df.groupby("run_id")

    per_run = []
    for run_id, g in run_groups:
        ep_returns = g["episodic_return"].to_numpy(dtype=float)
        if len(ep_returns) > 1:
            intra_reward_var = np.var(ep_returns, ddof=1)
        else:
            intra_reward_var = np.nan  # not enough episodes

        reward_mean = float(np.mean(ep_returns))

        dur_rates = g["duration_rate"].to_numpy(dtype=float)
        duration_rate_mean = float(np.mean(dur_rates))

        per_run.append(
            {
                "run_id": run_id,
                "intra_reward_var": intra_reward_var,
                "reward_mean": reward_mean,
                "duration_rate_mean": duration_rate_mean,
            }
        )

    per_run_df = pd.DataFrame(per_run)

    # Intra-agent metrics (across runs)
    intra_reward_var_mean = per_run_df["intra_reward_var"].mean()
    intra_reward_var_std = per_run_df["intra_reward_var"].std(ddof=1)

    intra_duration_rate_mean_mean = per_run_df["duration_rate_mean"].mean()
    intra_duration_rate_mean_std = per_run_df["duration_rate_mean"].std(ddof=1)

    # Inter-agent metrics (variance/std over run-level means)
    inter_reward_var = per_run_df["reward_mean"].var(ddof=1)
    inter_reward_std = per_run_df["reward_mean"].std(ddof=1)

    inter_duration_rate_var = per_run_df["duration_rate_mean"].var(ddof=1)
    inter_duration_rate_std = per_run_df["duration_rate_mean"].std(ddof=1)

    group_size = len(per_run_df)

    return {
        "group_size": group_size,
        # intra reward variance (per-run variance, then averaged)
        "intra_reward_var_mean": intra_reward_var_mean,
        "intra_reward_var_std": intra_reward_var_std,
        # intra duration rate (per-run mean duration rate, then averaged)
        "intra_duration_rate_mean_mean": intra_duration_rate_mean_mean,
        "intra_duration_rate_mean_std": intra_duration_rate_mean_std,
        # inter reward stats
        "inter_reward_var": inter_reward_var,
        "inter_reward_std": inter_reward_std,
        # inter duration rate stats
        "inter_duration_rate_var": inter_duration_rate_var,
        "inter_duration_rate_std": inter_duration_rate_std,
    }


def make_intra_plots(summary_df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # label = env + project for x-axis; tweak if you want exp_name in there too
    def label_row(row):
        return f"{row['env_id']}\n{row['project']}"

    summary_df = summary_df.copy()
    summary_df["group_label"] = summary_df.apply(label_row, axis=1)

    x = np.arange(len(summary_df["group_label"]))

    # 1) Intra-agent reward variance mean ± std
    plt.figure()
    plt.errorbar(
        x,
        summary_df["intra_reward_var_mean"],
        yerr=summary_df["intra_reward_var_std"],
        fmt="o",
    )
    plt.xticks(x, summary_df["group_label"], rotation=45, ha="right")
    plt.ylabel("Intra-agent reward variance\n(mean ± std across runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "intra_reward_variance.png"))
    plt.close()

    # 2) Intra-agent duration rate mean ± std
    plt.figure()
    plt.errorbar(
        x,
        summary_df["intra_duration_rate_mean_mean"],
        yerr=summary_df["intra_duration_rate_mean_std"],
        fmt="o",
    )
    plt.xticks(x, summary_df["group_label"], rotation=45, ha="right")
    plt.ylabel("Intra-agent mean duration rate\n(mean ± std across runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "intra_duration_rate.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to CSV produced by evaluate_wandb_policies.py",
    )
    parser.add_argument(
        "--output-summary-csv",
        required=True,
        help="Where to write group summary metrics CSV",
    )
    parser.add_argument(
        "--plot-dir",
        required=True,
        help="Directory (created if needed) for intra-agent plots",
    )
    parser.add_argument(
        "--group-cols",
        type=str,
        default="project,algo,env_id",
        help="Comma-separated columns defining a group "
             "(default: project,algo,env_id)",
    )
    args = parser.parse_args()

    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]

    df = pd.read_csv(args.input_csv)

    # ensure required columns
    required_cols = {"run_id", "episodic_return", "episode_length"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    if "duration_rate" not in df.columns:
        if "max_episode_steps" in df.columns:
            df["duration_rate"] = df["episode_length"] / df["max_episode_steps"]
        else:
            raise ValueError(
                "CSV has no duration_rate nor max_episode_steps. "
                "Rerun evaluation with duration_rate logging."
            )

    summary_rows = []

    for group_vals, g in df.groupby(group_cols):
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
        group_key = dict(zip(group_cols, group_vals))

        metrics = compute_group_metrics(g)
        row = {**group_key, **metrics}
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # save summary CSV
    os.makedirs(os.path.dirname(args.output_summary_csv) or ".", exist_ok=True)
    summary_df.to_csv(args.output_summary_csv, index=False)

    # plots
    make_intra_plots(summary_df, args.plot_dir)

    print(f"[DONE] Wrote summary to {args.output_summary_csv}")
    print(f"[DONE] Saved intra-agent plots to {args.plot_dir}")
    print()
    print("Inter-agent metrics (with group sizes):")
    cols_to_show = [
        c for c in [
            "project", "algo", "env_id",
            "group_size",
            "inter_reward_var", "inter_reward_std",
            "inter_duration_rate_var", "inter_duration_rate_std",
        ]
        if c in summary_df.columns
    ]
    print(summary_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
    