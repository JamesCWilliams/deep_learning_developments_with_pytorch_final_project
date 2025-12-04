#!/usr/bin/env python
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_metrics_for_config(df: pd.DataFrame):
    """
    df: episodes for a single config *after* alignment (only common seeds, aligned episode counts).
    Returns dict of intra- and inter-agent metrics + group_size (number of seeds).
    """
    per_run = []
    for (seed, run_id), g in df.groupby(["seed", "run_id"]):
        ep_returns = g["episodic_return"].to_numpy(dtype=float)
        dur_rates = g["duration_rate"].to_numpy(dtype=float)

        if len(ep_returns) > 1:
            intra_reward_var = np.var(ep_returns, ddof=1)
        else:
            intra_reward_var = np.nan

        reward_mean = float(np.mean(ep_returns))
        duration_rate_mean = float(np.mean(dur_rates))

        per_run.append(
            {
                "seed": seed,
                "run_id": run_id,
                "intra_reward_var": intra_reward_var,
                "reward_mean": reward_mean,
                "duration_rate_mean": duration_rate_mean,
            }
        )

    per_run_df = pd.DataFrame(per_run)
    group_size = len(per_run_df)

    # intra-agent stats (across runs)
    reward_intra_var_mean = per_run_df["intra_reward_var"].mean()
    reward_intra_var_std = per_run_df["intra_reward_var"].std(ddof=1)

    duration_intra_mean_mean = per_run_df["duration_rate_mean"].mean()
    duration_intra_mean_std = per_run_df["duration_rate_mean"].std(ddof=1)

    # inter-agent stats: mean±std over run-level means
    reward_inter_mean = per_run_df["reward_mean"].mean()
    reward_inter_std = per_run_df["reward_mean"].std(ddof=1)

    duration_inter_mean = per_run_df["duration_rate_mean"].mean()
    duration_inter_std = per_run_df["duration_rate_mean"].std(ddof=1)

    return {
        "group_size": group_size,
        "reward_intra_var_mean": reward_intra_var_mean,
        "reward_intra_var_std": reward_intra_var_std,
        "duration_intra_mean_mean": duration_intra_mean_mean,
        "duration_intra_mean_std": duration_intra_mean_std,
        "reward_inter_mean": reward_inter_mean,
        "reward_inter_std": reward_inter_std,
        "duration_inter_mean": duration_inter_mean,
        "duration_inter_std": duration_inter_std,
    }


def align_three_configs(dfs, labels):
    """
    dfs: list of 3 dataframes (one per config)
    labels: corresponding config labels (strings)

    Returns:
      - aligned_df: a single dataframe with 'config' column, only common seeds,
                    and episodes truncated per seed to min episode count across configs.
      - algo, env_id: common algo/env_id used (for sanity).
    """
    # attach config labels
    for df, label in zip(dfs, labels):
        df["config"] = label

    # sanity: same algo & env_id across all
    algos = {a for df in dfs for a in df["algo"].unique()}
    envs = {e for df in dfs for e in df["env_id"].unique()}
    if len(algos) != 1 or len(envs) != 1:
        raise ValueError(f"Expected same algo/env across all CSVs, got algos={algos}, envs={envs}")
    algo = next(iter(algos))
    env_id = next(iter(envs))

    # concat everything
    all_df = pd.concat(dfs, ignore_index=True)

    # ensure duration_rate exists
    if "duration_rate" not in all_df.columns:
        if "max_episode_steps" in all_df.columns:
            all_df["duration_rate"] = all_df["episode_length"] / all_df["max_episode_steps"]
        else:
            raise ValueError("CSV missing duration_rate and max_episode_steps; rerun evaluation.")

    # 1) align seeds: keep only seeds that appear in all configs
    seeds_per_config = {
        label: set(df["seed"].unique()) for df, label in zip(dfs, labels)
    }
    common_seeds = set.intersection(*seeds_per_config.values())
    all_df = all_df[all_df["seed"].isin(common_seeds)].copy()

    if all_df.empty:
        raise ValueError("No common seeds between configs after seed alignment.")

    # 2) align episodes: for each seed, take min #episodes across configs
    eps_counts = (
        all_df.groupby(["config", "seed"])["episode_index"]
        .nunique()
        .reset_index(name="n_episodes")
    )

    min_eps = (
        eps_counts.groupby("seed")["n_episodes"]
        .min()
        .reset_index(name="min_n_episodes")
    )

    all_df = all_df.merge(min_eps, on="seed", how="left")
    all_df = all_df[all_df["episode_index"] < all_df["min_n_episodes"]].copy()

    return all_df, algo, env_id


def make_intra_plots(summary_df: pd.DataFrame, algo: str, env_id: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    configs = summary_df["config"].tolist()
    x = np.arange(len(configs))

    # 1) Intra-agent reward variance
    plt.figure()
    plt.errorbar(
        x,
        summary_df["reward_intra_var_mean"],
        yerr=summary_df["reward_intra_var_std"],
        fmt="o",
    )
    plt.xticks(x, configs)
    plt.xlabel("Config")
    plt.ylabel("Intra-agent reward variance\n(mean ± std across seeds)")
    plt.title(f"{algo} on {env_id} — Intra-agent reward variance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{algo}_{env_id}_intra_reward_variance.png"))
    plt.close()

    # 2) Intra-agent duration rate
    plt.figure()
    plt.errorbar(
        x,
        summary_df["duration_intra_mean_mean"],
        yerr=summary_df["duration_intra_mean_std"],
        fmt="o",
    )
    plt.xticks(x, configs)
    plt.xlabel("Config")
    plt.ylabel("Intra-agent mean duration rate\n(mean ± std across seeds)")
    plt.title(f"{algo} on {env_id} — Intra-agent duration rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{algo}_{env_id}_intra_duration_rate.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csvs",
        nargs=3,
        required=True,
        help="Three CSV files to compare (same algo+env, different configs).",
    )
    parser.add_argument(
        "--labels",
        nargs=3,
        required=True,
        help="Three labels for the configs (e.g. baseline ga1 ga2).",
    )
    parser.add_argument(
        "--output-summary-csv",
        required=True,
        help="Where to write the summary metrics CSV (one row per config).",
    )
    parser.add_argument(
        "--plot-dir",
        required=True,
        help="Directory for intra-agent comparison plots.",
    )
    args = parser.parse_args()

    dfs = [pd.read_csv(p) for p in args.csvs]

    # basic sanity checks
    for df in dfs:
        for c in ["algo", "env_id", "seed", "run_id", "episode_index", "episodic_return", "episode_length"]:
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' missing in one of the CSVs.")

    aligned_df, algo, env_id = align_three_configs(dfs, args.labels)

    summary_rows = []
    for config_label in args.labels:
        cfg_df = aligned_df[aligned_df["config"] == config_label].copy()
        metrics = compute_metrics_for_config(cfg_df)
        metrics["config"] = config_label
        metrics["algo"] = algo
        metrics["env_id"] = env_id
        summary_rows.append(metrics)

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(args.output_summary_csv) or ".", exist_ok=True)
    summary_df.to_csv(args.output_summary_csv, index=False)

    make_intra_plots(summary_df, algo, env_id, args.plot_dir)

    print(f"[DONE] Summary written to {args.output_summary_csv}")
    print(f"[DONE] Plots written to {args.plot_dir}")
    print()
    print("Inter-agent metrics (means ± std across seeds):")
    print(
        summary_df[
            [
                "config",
                "group_size",
                "reward_inter_mean",
                "reward_inter_std",
                "duration_inter_mean",
                "duration_inter_std",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
