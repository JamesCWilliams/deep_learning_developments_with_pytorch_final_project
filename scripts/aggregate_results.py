from pathlib import Path
import pandas as pd


def main():
    here = Path.cwd()
    results = here / 'eval_results'
    combined_df = None

    for file in results.iterdir():
        name = str(file.stem)
        name_components = name.split('_')
        alg, env, weights = name_components[0], name_components[1], name_components[2]
        df = pd.read_csv(file)
        df_clean = df.drop(['project', 'run_id', 'run_name', 'exp_name', 'total_timesteps'], axis=1).reset_index()
        df_clean['weights'] = weights

        if combined_df is None:
            combined_df = df_clean
            continue
        else:
            combined_df = pd.concat([combined_df, df_clean], ignore_index=True)

    if combined_df is not None:
        combined_df = combined_df.drop('index', axis=1)
        combined_df.to_csv(str(here / 'aggregated_results.csv'))
    else:
        raise ValueError(f'No results found in {str(results)}. Make sure you run the script from the repo root.')


if __name__ == '__main__':
    main()
