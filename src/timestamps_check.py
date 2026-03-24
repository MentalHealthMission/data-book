import csv
import datetime
import os
import time

import numpy as np
import pandas as pd

from helper_funcs import convert_to_unix_time, df_filter
from timestamps_check_df_adjustment import df_adjustment


def check_all_data_types(
    files_list: list[str],
    EAS_threshold: float,
    timegap_threshold: float,
    measurement_cols: list[str],
    timestamp_col: str,
    end_time_col: str | None,
    duration_col: str | None,
    convert_to_unix: list | None,
    filter_dict: dict | None,
    df_adjustment_args: list | None,
    output_folder: str,
    site_col: str | None,
    participant_ID_col: str | None,
):
    """
    Calculates summary statistics for time stamp errors across all files in files_list.
    Also outputs two additional csvs to 'output_folder' that give more in depth information 
    """
    # Create an empty list to record number of errors for each participant for this data type.
    participants = []
    # Create an empty df to record all examples of errors for this data type
    examples = pd.DataFrame()

    for path in files_list:

        try:
            if path[-3:] == "csv":
                df = pd.read_csv(path)
            if path[-3:] == ".gz":
                df = pd.read_csv(path, compression="gzip")
        except:
            print(path + " file cannot be read")
            continue
        df = df_filter(df, filter_dict)
        df = df_adjustment(df, df_adjustment_args)
        if convert_to_unix is not None:
            df = convert_to_unix_time(df, convert_to_unix)
        examples, totals = counting_errors(
            df,
            EAS_threshold,
            timegap_threshold,
            measurement_cols,
            timestamp_col,
            end_time_col,
            duration_col,
            examples,
            site_col,
            participant_ID_col,
        )
        if totals[0] > 0:
            participants.append([path] + totals + [x / totals[0] for x in totals[1:7]])

    all_errors, max_fractions = update_output_array(participants)
    # Save output files for this data type
    # TODO find a way to tidy outputs
    write_examples_to_csv(examples, output_folder)
    write_participants_to_csv(participants,output_folder)

    return all_errors, max_fractions


def counting_errors(
    df: pd.DataFrame,
    EAS_thresh: float,
    STG_thresh: float,
    measurement_cols: list[str],
    time_stamp_col: str,
    end_time_col: str | None,
    duration_col: str | None,
    examples: pd.DataFrame,
    site_col: str | None,
    participant_ID_col: str | None,
):
    """
    Counts the number of timestamp errors in the input data frame 'df' 
    Returns:
        total_errors (list): a list of the total number of timestamps and the total number of each type of error.
        examples (pd.DataFrame): An updated list of examples of pairs of records with a timestamp error.
    """
    # TODO it should be optional to produce an examples dataframe.

    # Create empty list for recording totals of each error type (and total unique timestamps)
    total_errors = []
    # Clean df
    df, time_stamp_col = clean_df(
        measurement_cols, df, time_stamp_col, site_col, participant_ID_col
    )
    # Calculate number of timestamps and repeated timestamps and remove duplicates from df.
    total_errors, df = count_timestamps_and_RTs(total_errors, df, time_stamp_col)
    # Sort df and add columns used to count remaining errors
    df = sort_df(end_time_col, duration_col, df, measurement_cols, time_stamp_col)
    # Calculate the STG errors and get examples for STG and RT+CM errors
    total_errors, examples = STG_errors_and_examples(
        total_errors, examples, df, STG_thresh
    )
    # Get totals and examples for overlap errors
    total_errors, examples = count_EAS_errors(
        df,
        total_errors,
        end_time_col,
        duration_col,
        EAS_thresh,
        examples,
        time_stamp_col,
        STG_thresh,
    )

    return examples, total_errors


def update_examples(
    df: pd.DataFrame, indices: list, examples: pd.DataFrame, issue_type: str
):
    """
    Updates the dataframe 'examples' that contains pairs of records along with the type of error
    found in that pair. This function adds new examples that are found in the input 'df' at row
    numbers given in the input 'indices'. The examples all have the error described by the input
    string 'issue_type'.
    Returns:
        examples (pd.DataFrame): An updated version including new examples.
    """
    # Get a df of all top rows out of all example pairs
    shift_indices = [x - 1 for x in indices]
    df_v2 = df.add_suffix("_v2")
    new_examples_top_row = df_v2.iloc[shift_indices].copy()
    # Get df of all bottom rows out of all example pairs
    new_examples_bottom_row = df.iloc[indices].copy()
    # Create new_examples by concatenating above dfs sideways and adding a column for issue type
    new_examples_top_row = new_examples_top_row.reset_index(drop=True)
    new_examples_bottom_row = new_examples_bottom_row.reset_index(drop=True)
    issue_type_list = [issue_type] * len(new_examples_bottom_row)
    new_examples = pd.concat([new_examples_top_row, new_examples_bottom_row], axis=1)
    new_examples["issue"] = issue_type_list
    # Add new_examples to examples if it does not make the total length of examples too long (using 50000 for now)
    if (len(new_examples) + len(examples)) < 50000:
        examples = pd.concat([examples, new_examples], ignore_index=True)

    return examples


def STG_errors_and_examples(
    total_errors: list,
    examples: pd.DataFrame,
    df: pd.DataFrame,
    STG_thresh: float | int,
):
    """
    Get totals of STG errors (defined using 'STG_thresh') in 'df' and record in 'total_errors'.
    Update 'examples' with all instances of STG and RT+CM errors that are present in 'df'.
    Returns:
        total_errors (list): A list of the total number of all error types, updated to include STG errors
        examples (pd.DataFrame): A df containing examples of error types, updated with STG and RT+CM errorrs
    """
    # Count errors and update examples for STG-CM
    indices = df[
        (df["gap"] > 0) & (df["gap"] < STG_thresh) & (df["values_different"] == False)
    ].index.tolist()
    total_errors.append(len(indices))
    examples = update_examples(
        df,
        indices,
        examples,
        "small gap between records without measurement/duration change",
    )
    # Count errors and update examples for STG+CM
    indices = df[
        (df["gap"] > 0) & (df["gap"] < STG_thresh) & (df["values_different"])
    ].index.tolist()
    total_errors.append(len(indices))
    examples = update_examples(
        df,
        indices,
        examples,
        "small gap between records with measurement/duration change",
    )
    # Update examples for RT+CM (total number was calculated previously as we want number of unique timestamps)
    indices = df[(df["gap"] == 0) & (df["values_different"])].index.tolist()
    examples = update_examples(
        df, indices, examples, "no gap between records with measurement/duration change"
    )

    return total_errors, examples


def sort_df(
    end_time_col: str,
    duration_col: str,
    df: pd.DataFrame,
    measurement_cols: list[str],
    time_stamp_col: str,
):
    """
    Sorts the input df based on time_stamp_col and then either end_time_col or duration_col if 
    either of them are not None. Also adds columns for calculating timestamp errors using 
    time_stamp_col and measurement_cols
    """
    # Looks for a column for end time, retrieves end_time_col and sorts df on timestamp/end-time if one is found
    if end_time_col != None:
        df = df.sort_values(by=[time_stamp_col, end_time_col])
        df = df.reset_index(drop=True)
    else:
        # Looks for a column for duration, retrieves duration_col and sorts df on timestamp/duration if one is found
        if duration_col != None:
            df = df.sort_values(by=[time_stamp_col, duration_col])
            df = df.reset_index(drop=True)
        else:
            # If neither columns are found, sort solely on timestamp
            df = df.sort_values(by=time_stamp_col)
            df = df.reset_index(drop=True)

    # Add new column to be used to count errors
    df["gap"] = df[time_stamp_col] - df[time_stamp_col].shift()
    df["values_different"] = (df[measurement_cols] != df[measurement_cols].shift()).any(
        axis=1
    )

    return df


def count_EAS_errors(
    df: pd.DataFrame,
    total_errors: list,
    end_time_col: str,
    duration_col: str,
    EAS_thresh: float,
    examples: pd.DataFrame,
    time_stamp_col: str,
    STG: float,
):
    """
    Calculates the number of EAS errors if end_time_col or duration_col is not None
    Returns:
        total_errors (list): A list of the total number of all error types, updated to include EAS errors
        examples (df.DataFrame): A df containing examples of error types, updated with EAS errorrs
    """
    # calculate EAS errors and examples using end time if it exists
    if end_time_col != None:
        indices = df[
            (df[end_time_col].shift() > df[time_stamp_col]) & (df["gap"] > STG)
        ].index.tolist()
        total_errors.append(len(indices))
        indices = df[
            ((df[end_time_col].shift() - df[time_stamp_col]) > EAS_thresh)
            & (df["gap"] > STG)
        ].index.tolist()
        total_errors.append(len(indices))
        examples = update_examples(df, indices, examples, "overlapping records")
    else:
        # Calculate EAS errors and examples using duration if it exists
        if duration_col != None:
            df["calc_end_time"] = df[time_stamp_col] + df[duration_col]
            indices = df[
                (df["calc_end_time"].shift() > df[time_stamp_col]) & (df["gap"] > STG)
            ].index.tolist()
            total_errors.append(len(indices))
            indices = df[
                ((df["calc_end_time"].shift() - df[time_stamp_col]) > EAS_thresh)
                & (df["gap"] > STG)
            ].index.tolist()
            total_errors.append(len(indices))
            examples = update_examples(df, indices, examples, "overlapping records")
        else:
            # sets EAS errors to 0 if neither end time or duration exist
            total_errors.append(0)
            total_errors.append(0)

    return total_errors, examples


def count_timestamps_and_RTs(total_errors: list, df: pd.DataFrame, time_stamp_col: str):
    """
    Calculates the total number of unique timestamps and repeated timestamps (with and
    without changed measured values) in the input dataframe 'df' and adds these numbers
    to 'total_errors'. Also drops duplicate rows from 'df'.
    Returns:
        total_errors (list): An updated list of the total number of all error types.
        df (pd.DataFrame): Updated version of input 'df'; a dataframe for one data type for one participant.
    """
    # get total number of unique timestamps
    total_errors.append(df[time_stamp_col].nunique())
    # get RT-CM (total, not fraction)
    duplicate_rows = df[df.duplicated(keep=False)]
    total_errors.append(duplicate_rows[time_stamp_col].nunique())
    del duplicate_rows
    # get RT+CM
    df = df.drop_duplicates(keep="first")
    duplicates_subset = df[df.duplicated(subset=([time_stamp_col]), keep=False)]
    total_errors.append(duplicates_subset[time_stamp_col].nunique())
    del duplicates_subset

    return total_errors, df


def clean_df(
    measurement_cols: list[str],
    df: pd.DataFrame,
    time_stamp_col: str,
    site_col: str | None,
    participant_ID_col: str | None,
):
    """
    Cleans df so that it only includes relevant columns
    """
    if site_col is not None and participant_ID_col is not None:
        df = df.loc[
            :, (measurement_cols + [time_stamp_col, site_col, participant_ID_col])
        ].copy()
    if site_col is not None and participant_ID_col is None:
        df = df.loc[:, (measurement_cols + [time_stamp_col, site_col])].copy()
    if site_col is None and participant_ID_col is not None:
        df = df.loc[:, (measurement_cols + [time_stamp_col, participant_ID_col])].copy()
    if site_col is None and participant_ID_col is None:
        df = df.loc[:, (measurement_cols + [time_stamp_col])].copy()
    # TODO: check below line is ok
    pd.set_option("future.no_silent_downcasting", True)
    df = df.replace([None, "", pd.NA], "empty")

    return df, time_stamp_col


def update_output_array(participants: list[list]):
    """
    Returns a summary of how often each type of timestamp error occurs across all participants
    And a summary of the maximum amount of timestamp errors across all participants.
    """
    participants_np = (np.array(participants)[:, 1:14]).astype(float)
    all_errors_tots = np.sum(participants_np[:, 0:7], axis=0)
    all_errors = [all_errors_tots[0]] + (
        np.divide(all_errors_tots[1:7], all_errors_tots[0])
    ).tolist()
    max_fractions = [np.max(participants_np[:, 0])] + (
        np.max(participants_np[:, 7:13], axis=0)
    ).tolist()

    return all_errors, max_fractions


def write_participants_to_csv(participants, output_folder):
    """
    Writes the individual participant results stored in
    'participants' to a csv that will be saved in 'output folder'
    """
    column_names = [
        "participant",
        "total paired rows",
        "Number unexplained repeated timestep without measurement/duration change",
        "Number unexplained repeated timestep with measurement/duration change",
        "Number 0<time_gap<thresh without measurement/duration change",
        "Number 0<time_gap<thresh with measurement/duration change",
        "Number unexplained overlapping records",
        "Number unexplained overlapping records (over thresh)",
        "Fraction unexplained repeated timestep without measurement/duration change",
        "Fraction unexplained repeated timestep with measurement/duration change",
        "Fraction 0<time_gap<thresh without measurement/duration change",
        "Fraction 0<time_gap<thresh with measurement/duration change",
        "Fraction unexplained overlapping records",
        "Fraction unexplained overlapping records (over thresh)",
    ]
    participants.insert(0, column_names)
    if not os.path.exists(output_folder + "/timestamps errors by participant"):
        os.makedirs(output_folder + "/timestamps errors by participant")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csvname = (
        output_folder
        + "/timestamps errors by participant"
        + "/by_participant_"
        + current_time
    )
    with open(csvname + ".csv", "w", newline="") as f:
        write = csv.writer(f)
        write.writerows(participants)
    return


def write_examples_to_csv(examples, output_folder):
    """
    Writes timestamp error examples stored in the dataframe
    'examples' to a csv that will be saved in 'output folder'.
    """
    if not os.path.exists(output_folder + "/timestamps error examples"):
        os.makedirs(output_folder + "/timestamps error examples")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csvname = output_folder + "/timestamps error examples" + "/examples_" + current_time
    examples.to_csv(csvname + ".csv", index=False)

    return


def check_timestamp_errors(
    files_list: list[str],
    EAS_threshold: float,
    timegap_threshold: float,
    measurement_cols: list[str],
    timestamp_col="value.time",
    end_time_col=None,
    duration_col=None,
    convert_to_unix=False,
    filter_dict=None,
    df_adjustment_args=[None],
    output_folder="output_files",
    site_col=None,
    participant_ID_col=None,
):
    """
    This function performs timestamp checks on all files in files_list and outputs the results
    as a pandas dataframe
    """
    # Add any duration or end time cols to measurement cols (as we will also want to see if
    # these change when assessing RT and STG)
    if end_time_col != None:
        measurement_cols.append(end_time_col)
    if duration_col != None:
        measurement_cols.append(duration_col)

    all_errors, max_fractions = check_all_data_types(
        files_list,
        EAS_threshold,
        timegap_threshold,
        measurement_cols,
        timestamp_col,
        end_time_col,
        duration_col,
        convert_to_unix,
        filter_dict,
        df_adjustment_args,
        output_folder,
        site_col,
        participant_ID_col,
    )

    columns = [
        " ",
        "total counts",
        "fraction RT-CM",
        "fraction RT+CM",
        "fraction STG-CM",
        "fraction STG+CM",
        "fraction EAS",
        "fraction EAS (over thresh)",
    ]
    all_rows = [["All data"] + all_errors, ["Maximum"] + max_fractions]
    df = pd.DataFrame(all_rows, columns=columns)

    return df
