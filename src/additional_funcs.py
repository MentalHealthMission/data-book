import datetime
import time

import pandas as pd

from helper_funcs import convert_to_unix_time, df_filter


def investigate_sleep_blocks(
    files_list: list[str],
    timestamp_col: str,
    sleep_level_col: str,
    duration_col=None,
    end_time_col=None,
    convert_to_unix=None,
    filter_dict=None,
):
    """
    Returns a list of the length of each 'block' of sleep across all the files in files_list.
    """
    # TODO refactor this.
    # TODO addd documentation
    all_block_durations = []

    original_timestamp = timestamp_col
    for path in files_list:
        timestamp_col = original_timestamp

        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except:
            print(path + " file cannot be read")
            continue

        df = df_filter(df, filter_dict)
        if len(df) > 0:
            # convert to unix time if neccessary
            if convert_to_unix is not None:
                df = convert_to_unix_time(df, convert_to_unix)
            if end_time_col is not None:
                duration_col = "duration"
                df[duration_col] = df[end_time_col] - df[timestamp_col]

            df = df.sort_values(by=timestamp_col).reset_index(drop=True)
            # Clean EAS errors
            df["gap"] = df[timestamp_col].shift(-1) - df[timestamp_col]
            gaps = df[timestamp_col].diff().fillna(0)
            group_ids = (gaps > 0).cumsum()  # Start a new group when difference > 0
            df["group"] = group_ids
            df["gap"] = df.groupby("group")["gap"].transform("max")

            df = df[df[duration_col] > 0]
            df.loc[df[duration_col] > df["gap"], duration_col] = 0
            df["max duration"] = df.groupby("group")[duration_col].transform(
                "max"
            )  # get a column that is the max in the group
            df.loc[df[duration_col] < df["max duration"], duration_col] = 0
            # set all values less than the group max to zero
            sum_ = df.groupby("group")[duration_col].sum()
            df["sum"] = df["group"].map(sum_)
            df["use_datapoint"] = ~((df[duration_col] == 0) & (df["sum"] != 0))
            df[duration_col] = df.groupby("group")[duration_col].transform("max")
            df.loc[df[duration_col] == 0, duration_col] = df["gap"]
            mean_nonzero = (
                df[df["use_datapoint"]].groupby("group")[sleep_level_col].agg("first")
            )
            df[sleep_level_col] = df["group"].map(mean_nonzero)

            df = df.groupby("group", as_index=False).agg(
                {
                    duration_col: "first",
                    sleep_level_col: "first",
                    timestamp_col: "first",
                }
            )

            df["end"] = df[timestamp_col] + df[duration_col]
            df["time gap"] = df[timestamp_col] - df["end"].shift()
            df = df[df["time gap"] > 0]
            df["block duration"] = (
                df[timestamp_col].shift(-1)
                - df["time gap"].shift(-1)
                - df[timestamp_col]
            )
            df["block duration"] = df["block duration"] / 3600

            all_block_durations.extend(df["block duration"])

    return all_block_durations


def find_time_of_timestamps(all_file_paths, timestamp_col, convert_to_unix=None, filter_dict=None):
    """
    Returns a dictionary that reports how often each time of day occurs in the timestamp_col column 
    over all the files in all_file_paths
    """
    all_hours = []
    for path in all_file_paths:
        # read in file
        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except:
            print(path + " file cannot be read")
            continue
        df = df_filter(df, filter_dict)
        # convert to unix time if necessary
        if convert_to_unix is not None:
            df = convert_to_unix_time(df, convert_to_unix)
        # Add hour of timestamp to list 'all_hours'
        df["value.time.day"] = pd.to_datetime(
            df[timestamp_col], unit="s", origin="unix"
        )
        df["hour"] = df["value.time.day"].dt.strftime("%H:%M:%S")
        all_hours = all_hours + list(df["hour"])

    # Convert 'all_hours' into dictionary summarising how often each hour occurred
    d = dict.fromkeys(all_hours, 0)
    for val in all_hours:
        d[val] += 1
    return d


def time_gap_freqs(
    all_file_paths, output_path, time_stamp="value.time", filter_dict=None
):
    """
    Counts time gap frequecies.
    """
    all_data = pd.DataFrame()
    for path in all_file_paths:
        # read in file
        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except:
            print(path + " file cannot be read")
            continue

        df = df_filter(df, filter_dict)
        # get rid of all columns except timestamp
        df = df[[time_stamp]]
        df = df[~df[time_stamp].duplicated(keep="first")]
        df = df.sort_values(by=time_stamp).reset_index(drop=True)
        df["gap"] = df[time_stamp].diff().fillna(0)
        df = df[["gap"]]
        all_data = pd.concat([all_data, df], ignore_index=True)

    counts_df = all_data.value_counts().reset_index()
    counts_df["fraction"] = counts_df["count"] / len(all_data)
    counts_df.to_csv(output_path + "time_gaps.csv", index=True)
    df_first15 = counts_df.head(15)

    return df_first15
