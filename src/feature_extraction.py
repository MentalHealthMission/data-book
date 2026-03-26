import numpy as np
import pandas as pd

from clean_and_extract_features import get_timestamp_errors_and_clean
from helper_funcs import convert_to_unix_time, df_filter


def round_timestamp_to_midnight(df, timestamp_col):
    """
    Rounds all values in timestamp_col to the nearest midnight
    """
    df["value.time.day"] = pd.to_datetime(df[timestamp_col], unit="s", origin="unix")
    df["hour"] = df["value.time.day"].dt.hour + df["value.time.day"].dt.minute / 60
    df["hour"] = df["hour"].apply(lambda x: x - 24 if x > 12 else x)
    # Round timestamp to nearest midnight
    df["rounded"] = df["value.time.day"].apply(
        lambda ts: (ts + pd.Timedelta(minutes=720)).normalize()
    )
    df[timestamp_col] = df["rounded"].astype("int64") // 10**9
    cleaned_df, features = get_timestamp_errors_and_clean(
        df=df,
        interval="D",
        time_stamp_col=timestamp_col,
        measurement_col="hour",
        STG=86400,
        meas_agg="mean",
    )
    hours = get_fixed_series(
        cleaned_df, "D", "mean", "hour", timestamp_col, "hour of datapoint"
    )

    return df, cleaned_df, hours


def get_extra_HR_metadata_features(
    cleaned_df,
    timestamp_col,
    meas_col,
    max_gap,
    interval,
    low_thresh=30,
    upper_thresh=250,
    end_time_col=None,
    duration_col=None,
    included_errors=["RT+CM", "STG+CM", "STG-CM", "EAS"],
):
    """
    Produces extra metadata features for input dataframe cleaned_df
    """
    # Calculate filtered steps and get count
    cleaned_df["filtered"] = cleaned_df[meas_col].clip(
        lower=low_thresh, upper=upper_thresh
    )
    cleaned_df["Number filtered"] = (
        cleaned_df["filtered"] != cleaned_df[meas_col]
    ).astype(int)

    # Create a new column in df for total errors
    if (
        end_time_col is None and duration_col is None
    ):  # This is just in case EAS was in included_errors by mistake
        if "EAS" in included_errors:
            included_errors.remove("EAS")
    included_errors.append("Number filtered")
    cleaned_df["total timestamps with any error"] = cleaned_df[included_errors].max(
        axis=1
    )

    # Extract metadata features from df
    df_errors = cleaned_df.loc[
        :, included_errors + ["total timestamps with any error"]
    ].copy()
    df_errors["total counts"] = 1
    features = df_errors.resample(interval).sum()

    coverage_features = get_coverage(
        cleaned_df.copy(),
        timestamp_col,
        max_gap,
        interval,
        "Coverage (secs) from all datapoints",
        end_time_col,
        duration_col,
    )
    only_clean = cleaned_df[cleaned_df["total timestamps with any error"] == 0]
    coverage_features_filtered = get_coverage(
        only_clean.copy(),
        timestamp_col,
        max_gap,
        interval,
        "Coverage (secs) from clean datapoints",
        end_time_col,
        duration_col,
    )

    # merge all metadata
    metadata_features = pd.concat(
        [features, coverage_features, coverage_features_filtered], axis=1
    )
    metadata_features.index.name = cleaned_df.index.name

    return metadata_features, cleaned_df


def get_fixed_series(
    df,
    interval,
    agg,
    meas_col,
    timestamp_col,
    new_name,
    extended_index=None,
):
    """
    Takes in an input df and produces an output df that contains a resampled series of the
    field meas_col with the specified interval, resampling method defined by agg. The name
    of the field in the output df is set by new_name, and is extended by extended_index if
    it is set.
    Returns:
        output_series: a df containing a resampled series for a field of the input df.
    """

    df["value.time.day"] = pd.to_datetime(df[timestamp_col], unit="s", origin="unix")
    timestamp_col = "value.time.day"
    index_to_drop = df[
        df[timestamp_col].dt.year == 1970
    ].index  # Remove rows where the year is 1970, which indicates no data was recorded
    df.drop(index_to_drop, inplace=True)
    df.set_index(timestamp_col, inplace=True)

    # Carry out the resampling
    df = df.loc[:, [meas_col]].copy()
    if agg == "count":
        output_series = df.resample(interval).count()
    if agg == "max":
        output_series = df.resample(interval).max()
    if agg == "min":
        output_series = df.resample(interval).min()
    if agg == "sum":
        output_series = df.resample(interval).sum()
    if agg == "mean":
        output_series = df.resample(interval).mean()

    # reindex the output series so it is the required length
    if extended_index is not None:
        output_series = output_series.reindex(extended_index).fillna(0)

    # rename the measurement column
    output_series.rename(columns={meas_col: new_name}, inplace=True)

    return output_series


def find_durations(df, start_col, end_col, interval, meas_col=None):
    """
    The incoming df should already be cleaned such that the values in end_col are always after
    the values in start_col
    returns a version of df with end and start cols converted to datetime objects and extra column
    'seconds_diff' which gives the duration of each datapoint. Any datapoints overlapping the hour
    have been split into separate datapoints
    meas col is a numerical measurement column that will be split proportionally between the two new datapoints
    """
    if meas_col is not None:
        df["duration"] = df[end_col] - df[start_col]
    # Convert df to right format for resampling
    df[start_col] = pd.to_datetime(df[start_col], unit="s", origin="unix")
    df[end_col] = pd.to_datetime(df[end_col], unit="s", origin="unix")
    df = split_intervals(df, interval, start_col, end_col)
    df["seconds_diff"] = (df[end_col] - df[start_col]).dt.total_seconds()
    if meas_col is not None:
        df[meas_col] = df[meas_col] * df["seconds_diff"] / df["duration"]

    return df


def split_intervals(df, freq, start_col, end_col):
    df = df.copy()
    # Create interval boundaries per row
    df["boundary"] = df.apply(
        lambda r: pd.date_range(
            r[start_col].floor(freq), r[end_col].ceil(freq), freq=freq
        ),
        axis=1,
    )
    # Explode
    df = df.explode("boundary", ignore_index=True)
    # Compute new start
    df["new_start"] = df[[start_col, "boundary"]].max(axis=1)
    # Compute next boundary
    df["next_boundary"] = df["boundary"] + pd.tseries.frequencies.to_offset(freq)
    # Compute new end
    df["new_end"] = df[[end_col, "next_boundary"]].min(axis=1)
    # Keep only valid intervals
    df = df[df["new_start"] < df["new_end"]]
    # Clean up
    df = df.drop(columns=[start_col, end_col, "boundary", "next_boundary"])
    df = df.rename(columns={"new_start": start_col, "new_end": end_col})

    return df.reset_index(drop=True)


def weighted_average(
    df,
    timestamp_col,
    meas_col,
    max_time_gap,
    interval,
    col_name,
    end_time_col=None,
    duration_col=None,
):
    """
    Returns a df that reports the weighted average of meas_col for the input interval
    """
    if duration_col is not None:
        df["end_time"] = df[timestamp_col] + df[duration_col]
        df = find_durations(df, timestamp_col, "end_time_col", interval)
    if end_time_col is None and duration_col is None:
        df["start_time_1"] = df[timestamp_col] - (max_time_gap / 2)
        df["start_time_2"] = 0.5 * (
            df[timestamp_col] + df[timestamp_col].shift().fillna(0)
        )
        df["start_time_col"] = df[["start_time_1", "start_time_2"]].max(axis=1)
        df["end_time_1"] = df[timestamp_col] + (max_time_gap / 2)
        df["end_time_2"] = 0.5 * (
            df[timestamp_col] + df[timestamp_col].shift(-1).fillna(100000000000000)
        )
        df["end_time_col"] = df[["end_time_1", "end_time_2"]].min(axis=1)
        df = find_durations(df, "start_time_col", "end_time_col", interval)
        timestamp_col = "start_time_col"  # Change this so the start time column is used in get_fixed_series_below
    if duration_col is None and end_time_col is not None:
        df = find_durations(df, timestamp_col, end_time_col, interval)
    duration_col = "seconds_diff"
    df["weighted"] = df[meas_col] * df[duration_col]
    total_duration = get_fixed_series(
        df, interval, "sum", duration_col, timestamp_col, "total_duration"
    )  # TODO consider also using this as a metadata feature
    total_weighted = get_fixed_series(
        df, interval, "sum", "weighted", timestamp_col, "total_weighted"
    )
    final_df = pd.concat([total_duration, total_weighted], axis=1)
    final_df[col_name] = final_df["total_weighted"] / final_df["total_duration"]
    final_df[col_name] = np.where(
        final_df["total_duration"] == 0,
        -1,
        final_df["total_weighted"] / final_df["total_duration"],
    )
    final_df = final_df[[col_name]]

    return final_df


def get_coverage(
    df,
    timestamp_col,
    max_time_gap,
    interval,
    col_name,
    end_time_col=None,
    duration_col=None,
):
    """
    Returns a dataframe that reports the amount of time covered each interval
    """

    if duration_col is not None:
        df["end_time"] = df[timestamp_col] + df[duration_col]
        df = find_durations(df, timestamp_col, "end_time_col", interval)
    if end_time_col is None and duration_col is None:
        df["start_time_1"] = df[timestamp_col] - (max_time_gap / 2)
        df["start_time_2"] = 0.5 * (
            df[timestamp_col] + df[timestamp_col].shift().fillna(0)
        )
        df["start_time_col"] = df[["start_time_1", "start_time_2"]].max(axis=1)
        df["end_time_1"] = df[timestamp_col] + (max_time_gap / 2)
        df["end_time_2"] = 0.5 * (
            df[timestamp_col] + df[timestamp_col].shift(-1).fillna(100000000000000)
        )
        df["end_time_col"] = df[["end_time_1", "end_time_2"]].min(axis=1)
        df = find_durations(df, "start_time_col", "end_time_col", interval)
        timestamp_col = "start_time_col"  # Change this so the start time column is used in get_fixed_series_below
    if duration_col is None and end_time_col is not None:
        df = find_durations(df, timestamp_col, end_time_col, interval)
    duration_col = "seconds_diff"
    total_duration = get_fixed_series(
        df, interval, "sum", duration_col, timestamp_col, col_name
    )  # TODO consider also using this as a metadata feature

    return total_duration


def get_sleep_features(
    df,
    timestamp_col,
    sleep_stage_col,
    awake_string,
    sleep_stages,
    interval,
    end_time_col=None,
    duration_col=None,
    filter_dict=None,
):

    df = df_filter(df, filter_dict)

    if (
        end_time_col is None
    ):  # either end col or duration col should not be None for sleep data
        df["end_time"] = df[timestamp_col] + df[duration_col]
        end_time_col = "end_time"
    if duration_col is None:
        df["duration"] = df[end_time_col] - df[timestamp_col]
        duration_col = "duration"

    # TODO need to think of better way to do below
    df_for_TST = df[df[sleep_stage_col].isin(sleep_stages)].copy()
    df_for_TST = find_durations(df_for_TST, "value.time", end_time_col, "D")
    TST_ = get_fixed_series(
        df_for_TST,
        interval,
        "sum",
        "seconds_diff",
        timestamp_col,
        "total sleep duration",
    )

    df = df.sort_values(timestamp_col)

    # Identify breaks between blocks
    df["new_block"] = df[timestamp_col] != df[end_time_col].shift()
    # Assign block IDs
    df["block_id"] = df["new_block"].cumsum()

    df["awake_duration"] = df[duration_col].where(
        df[sleep_stage_col] == awake_string, 0
    )
    df["awake_count"] = (df[sleep_stage_col] == awake_string).astype(int)

    df[end_time_col] = pd.to_datetime(df[end_time_col], unit="s", origin="unix")

    df = df.groupby("block_id", as_index=False).agg(
        {
            end_time_col: "last",
            duration_col: "sum",
            "awake_count": "sum",
            "awake_duration": "sum",
        }
    )

    # use fixed series to get the count of blocks in a day and the biggest duration in a day
    number_episodes = get_fixed_series(
        df,
        interval,
        "count",
        duration_col,
        end_time_col,
        "number of sleep episodes",
    )
    max_episodes = get_fixed_series(
        df,
        interval,
        "max",
        duration_col,
        end_time_col,
        "longest sleep episode duration",
    )
    df_max = df.groupby(df[end_time_col].dt.date, group_keys=False).apply(
        lambda x: x.loc[x[duration_col].idxmax()]
    )  # make sure this works properly - need to make sure it always choses same row

    # This should leave just the biggest block each day, now use fixed series to get sum of awake_duration and awake_count
    number_awakenings = get_fixed_series(
        df_max,
        interval,
        "sum",
        "awake_count",
        end_time_col,
        "number of awake stages during longest sleep episode",
    )
    duration_awakenings = get_fixed_series(
        df_max,
        interval,
        "sum",
        "awake_duration",
        end_time_col,
        "total duration of awake stages during longest sleep episode",
    )

    df_max["wake_up_time"] = (
        df_max[end_time_col].dt.hour * 3600
        + df_max[end_time_col].dt.minute * 60
        + df_max[end_time_col].dt.second
    )
    df_max["sleep_onset_time"] = df_max["wake_up_time"] - df_max[duration_col]

    wake_ups = get_fixed_series(
        df_max,
        interval,
        "sum",
        "wake_up_time",
        end_time_col,
        "End time of longest sleep episode (In seconds from midnight of current day)",
    )
    sleep_onset = get_fixed_series(
        df_max,
        interval,
        "sum",
        "sleep_onset_time",
        end_time_col,
        "start time of longest sleep episode (In seconds from midnight of current day)",
    )
    all_features = pd.concat(
        [
            TST_,
            number_episodes,
            max_episodes,
            number_awakenings,
            duration_awakenings,
            wake_ups,
            sleep_onset,
        ],
        axis=1,
    )

    return all_features


def general_steps_cleaning_and_FE(
    df,
    interval,
    meas_col,
    timestamp_col,
    STG,
    EAS_thresh,
    convert_to_unix=True,
    meas_agg="mean",
    duration_col=None,
    SPS=4,
    filter_min=50,
    end_time_col=None,
    STG_fix=False,
    cumulative=False,
    device_col=None,
    filter_dict=None,
    round_to_midnight=False,
    distribute_steps=False,
    included_errors=["RT+CM", "STG+CM", "STG-CM"],
):
    # TODO change to unix time straight way if not unix - or have a general assumption of this data book be that everything should be in unix time
    # TODO - consider deleting STG-CM column at the end if cumulative as we are setting it to zero (can't delete it before)
    # TODO - check device fix works and add note about what you have done
    df = df.sort_values(by=timestamp_col)

    if round_to_midnight:
        # TODO think about whether below is correct
        df, df_hour_cleaned, hour_features = round_timestamp_to_midnight(
            df, timestamp_col
        )
        if end_time_col is not None:
            df[end_time_col] = df[timestamp_col] + 86400

    df = df_filter(df, filter_dict)
    # convert to unix time if neccessary
    if convert_to_unix is not None:
        df = convert_to_unix_time(df, convert_to_unix)

    counts_raw = get_fixed_series(
        df, interval, "count", meas_col, timestamp_col, "total raw datapoints"
    )
    extended_index = pd.date_range(
        start=counts_raw.index.min(), end=counts_raw.index.max(), freq=interval
    )

    if not cumulative:
        df = df[df[meas_col] > 0].copy()
        df = df.reset_index(drop=True)

    if cumulative and (device_col is not None):
        # Fix where transition happens if there is a single RT+CM or STG+CM
        df["dif_device"] = (df[[device_col]] != df[[device_col]].shift()).any(axis=1)
        df.iloc[0, df.columns.get_loc("dif_device")] = False
        df["time gap"] = df[timestamp_col].diff().fillna(0)
        mask = (
            df["dif_device"]
            & (df["time gap"] < STG)
            & df["dif_device"].shift(-1, fill_value=False)
            & (df["time gap"].shift(-1, fill_value=0) > STG)
            & df["dif_device"].shift(fill_value=False)
            & (df["time gap"].shift(fill_value=0) > STG)
        )
        prev_ts = df[timestamp_col].shift().fillna(0)
        df.loc[mask, timestamp_col] = prev_ts[mask] - STG
        df = df.sort_values(by=timestamp_col)
        df["dif_device"] = (df[[device_col]] != df[[device_col]].shift()).any(axis=1)
        df.iloc[0, df.columns.get_loc("dif_device")] = False
        df["time gap"] = df[timestamp_col].diff().fillna(0)
        mask = (
            df["dif_device"]
            & (df["time gap"] < STG)
            & ~df["dif_device"].shift(-1, fill_value=False)
            & (df["time gap"].shift(-1, fill_value=0) > STG)
            & ~df["dif_device"].shift(fill_value=False)
            & (df["time gap"].shift(fill_value=0) > STG)
        )
        prev_ts = df[timestamp_col].shift().fillna(0)
        df.loc[mask, timestamp_col] = prev_ts[mask] + STG
        # Produce a cleaned version that has a device col so that this can be merged later.
        df_clean_device, features = get_timestamp_errors_and_clean(
            df,
            interval,
            timestamp_col,
            device_col,
            STG,
            EAS_thresh,
            STG_fix=STG_fix,
            meas_agg="first",
            end_time_col=end_time_col,
            duration_col=duration_col,
        )
    if end_time_col is None and duration_col is None:
        df["previous_time_stamp"] = df[timestamp_col].shift().fillna(0)
        df_clean_pts, features = get_timestamp_errors_and_clean(
            df,
            interval,
            timestamp_col,
            "previous_time_stamp",
            STG,
            STG_fix=STG_fix,
            meas_agg="min",
        )

    cleaned_df, features = get_timestamp_errors_and_clean(
        df=df,
        interval=interval,
        time_stamp_col=timestamp_col,
        measurement_col=meas_col,
        EAS_thresh=EAS_thresh,
        STG=STG,
        meas_agg=meas_agg,
        duration_col=duration_col,
        end_time_col=end_time_col,
        STG_fix=STG_fix,
    )
    if end_time_col is None and duration_col is None:
        cleaned_df["previous_time_stamp"] = df_clean_pts["previous_time_stamp"]
    if cumulative and (device_col is not None):
        cleaned_df[device_col] = df_clean_device[device_col]
    if round_to_midnight:
        cleaned_df["hour"] = df_hour_cleaned["hour"]
    if cumulative:
        cleaned_df["new steps"] = cleaned_df[meas_col].diff().fillna(0)
        meas_col = "new steps"
        cleaned_df["STG-CM"] = 0
        cleaned_df["RT+CM"] = (
            (cleaned_df["RT+CM"].shift() == 1) | (cleaned_df["RT+CM"] == 1)
        ).astype(int)
        cleaned_df["STG+CM"] = (
            (cleaned_df["STG+CM"].shift() == 1) | (cleaned_df["STG+CM"] == 1)
        ).astype(int)
        if device_col is not None:
            cleaned_df["same_device"] = (
                cleaned_df[[device_col]] == cleaned_df[[device_col]].shift()
            ).any(axis=1)
            cleaned_df.iloc[0, cleaned_df.columns.get_loc("same_device")] = True
            cleaned_df = cleaned_df[cleaned_df["same_device"]].copy()
        cleaned_df = cleaned_df[cleaned_df[meas_col] > 0].copy()
    if duration_col is not None:  # TODO need to consider if end time=timestamp
        cleaned_df["filtered steps"] = cleaned_df[meas_col].clip(
            upper=SPS * cleaned_df[duration_col]
        )
    if end_time_col is not None:  # TODO need to consider if duration==0
        cleaned_df["filtered steps"] = cleaned_df[meas_col].clip(
            upper=SPS * (cleaned_df[end_time_col] - cleaned_df[timestamp_col])
        )
    if end_time_col is None and duration_col is None:
        cleaned_df["allowed steps"] = (
            cleaned_df[timestamp_col] - cleaned_df["previous_time_stamp"]
        ) * SPS
        cleaned_df["filtered steps"] = cleaned_df[["allowed steps", meas_col]].min(
            axis=1
        )
    cleaned_df["filtered steps"] = np.where(
        cleaned_df[meas_col] < filter_min,
        cleaned_df[meas_col],
        cleaned_df["filtered steps"].clip(lower=filter_min),
    )
    cleaned_df["Number filtered"] = (
        cleaned_df["filtered steps"] != cleaned_df[meas_col]
    ).astype(int)
    # Create a new column in df for total errors
    cleaned_df["total timestamps with any error"] = cleaned_df[
        included_errors + ["Number filtered"]
    ].max(axis=1)
    # Extract metadata features from df
    df_errors = cleaned_df.loc[
        :, included_errors + ["Number filtered", "total timestamps with any error"]
    ].copy()
    df_errors["total counts"] = 1
    features = df_errors.resample(interval).sum()
    features = features.reindex(extended_index).fillna(0)
    if round_to_midnight:
        features = pd.concat(
            [features, hour_features["hour of datapoint"], counts_raw], axis=1
        )
    else:
        features = pd.concat([features, counts_raw], axis=1)
    features.index.name = cleaned_df.index.name

    if distribute_steps:
        if end_time_col is None:
            end_time_col = "end_time_col"
            if duration_col is None:
                cleaned_df["end_time_col"] = cleaned_df[timestamp_col]
                cleaned_df[timestamp_col] = cleaned_df["previous_time_stamp"]
            else:
                cleaned_df["end_time_col"] = (
                    cleaned_df[timestamp_col] + cleaned_df[duration_col]
                )
        df_for_steps = cleaned_df.copy()
        df_for_filtered = cleaned_df.copy()
        cleaned_df_steps = find_durations(
            df_for_steps, timestamp_col, end_time_col, interval, meas_col
        )
        cleaned_df_filtered = find_durations(
            df_for_filtered, timestamp_col, end_time_col, interval, "filtered steps"
        )
        total_filtered = get_fixed_series(
            cleaned_df_filtered,
            interval,
            "sum",
            "filtered steps",
            timestamp_col,
            "Total steps (with filtering)",
        )
        total_unfiltered = get_fixed_series(
            cleaned_df_steps,
            interval,
            "sum",
            meas_col,
            timestamp_col,
            "Total steps (without filtering)",
        )
    else:
        total_filtered = get_fixed_series(
            cleaned_df,
            interval,
            "sum",
            "filtered steps",
            timestamp_col,
            "Total steps (with filtering)",
        )
        total_unfiltered = get_fixed_series(
            cleaned_df,
            interval,
            "sum",
            meas_col,
            timestamp_col,
            "Total steps (without filtering)",
        )

    total_steps = pd.concat([total_filtered, total_unfiltered], axis=1)
    total_steps = total_steps.reindex(extended_index).fillna(0)
    total_steps.index.name = cleaned_df.index.name

    return features, cleaned_df, total_steps
