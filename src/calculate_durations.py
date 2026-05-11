import statistics
from collections import Counter

import pandas as pd

from calculate_durations_df_adjustment import df_adjustment
from helper_funcs import convert_to_unix_time, df_filter


def check_all_data_types(
    file_list,
    thresh,
    timestamp_col,
    end_time_col,
    duration_col,
    convert_to_unix,
    filter_dict,
    df_adjustment_args,
):
    """
    Calculates summary statistics about the average time gap between records based on the
    timestamp_col and the duration of each datapoint (using end_time_col or duration_col if
    one of them is not None) for all data in all files in files_list. First converts the time
    columns to unix if convert_to_unix is not None. Also cleans the data according to the
    information in filter_dict and df_adjustment_args
    """
    # Create an empty list to record number of errors for each participant for this data type.
    gap_totals = []
    duration_totals = []

    for path in file_list:
        # Read in the csv as a df and check it for timestamp errors
        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except Exception as e:
            print(path + " file cannot be read, error: " + str(e))
            continue

        df = df_filter(df, filter_dict)
        # convert to unix time if neccessary
        if convert_to_unix is not None:
            df = convert_to_unix_time(df, convert_to_unix)
        df = df_adjustment(df, df_adjustment_args)
        gap_totals, duration_totals = getting_gaps(
            df, timestamp_col, end_time_col, duration_col, gap_totals, duration_totals
        )

    gap_summaries, duration_summaries = update_output_array(
        gap_totals, duration_totals, thresh
    )

    return gap_summaries, duration_summaries


def getting_gaps(
    df: pd.DataFrame,
    time_stamp_col: str,
    end_time_col: str,
    duration_col: str,
    gap_totals: list[int],
    duration_totals: list[int],
):
    """
    Records all the time gaps between consecutive records and durations of each record (where applicable)
    in the dataframe 'df'. Adds these to the lists gap_totals and duration_totals.
    Returns:
        gap_totals (list): a list of all the time gaps between records, updated for the current participant
        duration_totals (list): a list of all the record durations, updated for the current participant
    """

    # Remove duplicates
    df = df[~df[time_stamp_col].duplicated(keep="first")]
    df = df.copy()

    # Add all durations to duration totals if duration or end time is reported
    if end_time_col is not None:
        # TODO fix warning here
        df["duration"] = df[end_time_col] - df[time_stamp_col]
        duration_totals = duration_totals + df["duration"].tolist()[1:]
    if duration_col is not None:
        duration_totals = duration_totals + df[duration_col].tolist()[1:]

    # Add all time gaps to gap_totals
    df = df.sort_values(by=time_stamp_col)
    df["gap"] = df[time_stamp_col] - df[time_stamp_col].shift()
    gap_totals = gap_totals + df["gap"].tolist()[1:]

    return gap_totals, duration_totals


def update_output_array(gap_totals: list, duration_totals: list, thresh: float):
    """
    Produces summary stats from the gap_totals and durations_totals, some of which
    depend on the value of thresh
    """
    gap_summaries = summary_stats(gap_totals, thresh)
    if len(duration_totals) > 0:
        duration_summaries = summary_stats(duration_totals, thresh)
    else:
        duration_summaries = ["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"]

    return gap_summaries, duration_summaries


def summary_stats(LIST, thresh):
    """
    Calculates summary statistics for input 'LIST'
    Returns:
        summary_list (list): a list of summary data (e.g, max, median) from the input list
    """
    counts = Counter(LIST)
    mode, mode_count = counts.most_common(1)[0]
    less_than_mode = sum(1 for x in LIST if x < (mode - thresh))
    close_to_mode = sum(
        1 for x in LIST if (x >= (mode - thresh) and x <= (mode + thresh))
    )
    av = sum(LIST) / len(LIST)
    med = statistics.median(LIST)
    min_list = min(LIST)
    max_list = max(LIST)
    summary_list = [
        av,
        med,
        min_list,
        max_list,
        mode,
        mode_count / (len(LIST)),
        less_than_mode / (len(LIST)),
        close_to_mode / (len(LIST)),
    ]

    return summary_list


def investigate_frequency(
    files_list: list[str],
    thresh: float,
    timestamp_col="value.time",  # Name of timestamp column
    end_time_col=None,  # Name of end time column. Should either be a string or None.
    duration_col=None,  # Name of duration column. Should either be a string or None.
    convert_to_unix=None,
    filter_dict=None,
    df_adjustment_args=[None],
):
    """
    Calculates summary statistics about the average time gap between records based on the
    timestamp_col and the duration of each datapoint (using end_time_col or duration_col if
    one of them is not None) for all data in all files in files_list. First converts the time
    columns to unix if convert_to_unix is not None. Also cleans the data according to the
    information in filter_dict and df_adjustment_args. Concatenates these column statistics with
    the column names and returns an output df.
    """

    gaps, durations = check_all_data_types(
        files_list,
        thresh,
        timestamp_col,
        end_time_col,
        duration_col,
        convert_to_unix,
        filter_dict,
        df_adjustment_args,
    )

    columns = [
        " ",
        "Mean",
        "Median",
        "Min",
        "Max",
        "Mode",
        "number that are mode",
        "number under mode (" + str(thresh) + " sec buffer)",
        "number close to mode (within " + str(thresh) + " secs)",
    ]

    data = [["time gaps"] + gaps, ["durations"] + durations]
    df = pd.DataFrame(data, columns=columns)

    return df
