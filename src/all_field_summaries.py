import math
import statistics

import pandas as pd

from all_field_summaries_df_adjustment import df_adjustment
from helper_funcs import df_filter


def retrieve_all_data(new_field_names, files_list, timestamp, filter_dict, df_adjustment_args):
    """
    Returns a 2d array that contains all entries in each column listed in 'new_field_names'
    across every file in files_list. Each file is first cleaned using the information in
    filter_dict and df_adjustment_args. Duplicate entries (same values at the same time) are 
    deleted.
    """
    all_values = []
    for i in range(0, len(new_field_names)):
        all_values.append([])

    # Loop over all files
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
        for j in range(0, len(new_field_names)):
            if new_field_names[j] != timestamp:
                df_copy = df[[timestamp, new_field_names[j]]].copy()
            else:
                df_copy = df[[new_field_names[j]]].copy()

            df_cleaned = df_copy.drop_duplicates()
            all_values[j].extend(df_cleaned[new_field_names[j]])

    return all_values


def summary_stats(cleaned_list: list):
    """
    Returns descriptive statistics for cleaned_list
    """
    cleaned_list.sort()
    LQT_ind = int(len(cleaned_list) * 0.25)
    UQT_ind = int(len(cleaned_list) * 0.75)
    P1_ind = int(len(cleaned_list) * 0.01)
    P99_ind = int(len(cleaned_list) * 0.99)
    LQT = cleaned_list[LQT_ind]
    UQT = cleaned_list[UQT_ind]
    P1 = cleaned_list[P1_ind]
    P99 = cleaned_list[P99_ind]
    av = sum(cleaned_list) / len(cleaned_list)
    med = statistics.median(cleaned_list)
    min_list = min(cleaned_list)
    max_list = max(cleaned_list)

    return av, med, min_list, max_list, LQT, UQT, P1, P99


def count_instances(LIST: list):
    """
    This investigates how many times each item occurs in a list for every item that is present
    in the list. Returns the total number of unique items and either a string describing the
    composition of the list or the string 'too big' if there are more than 15 unique items in
    the list.
    """
    # create a dictionary from a list, the keys will be every item that occurs in the list
    # and the values will be how often that item occurs
    d = dict.fromkeys(LIST, 0)
    for val in LIST:
        d[val] += 1

    # Print the number of dictionary items and the whole dictionary unless it has
    # too many items (I used 15 as the cut off).
    length_d = len(d)
    if len(d) < 15:
        dictstring = str(d)
    if len(d) >= 15:
        dictstring = "Too big, " + str(len(d)) + " unique values"

    return length_d, dictstring


def get_row(new_field_names: list[str], all_values: list[list[str]]):
    """
    Produces summary statistics for the data type and appends a row for this data type to the
    output array (all_rows).
    """
    all_rows = []
    for k in range(0, len(new_field_names)):
        length_d, dictstring = count_instances(all_values[k])
        the_row = [new_field_names[k], len(all_values[k]), dictstring]
        if all(isinstance(item, float) for item in all_values[k]) or all(
            isinstance(item, int) for item in all_values[k]
        ):
            cleaned_list = [x for x in all_values[k] if not math.isnan(x)]
            if len(cleaned_list) > 0:
                av, med, minlist, maxlist, LQT, UQT, P1, P99 = summary_stats(
                    all_values[k]
                )
                the_row.extend([av, minlist, P1, LQT, med, UQT, P99, maxlist])

        else:
            the_row.extend(["N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
        all_rows.append(the_row)

    return all_rows


def Summarise_fields(
    files_list: list[str],
    fields: list[str],
    time_stamp="value.time",
    filter_dict=None,
    df_adjustment_args=[None],
):
    """
    Produces an output csv that reports summary statistics for all fields listed in fields.
    The summary statistics are reported using all entries in that field across all files in
    files_list. The data in each file is first cleaned using the information in filter_dict
    and df_adjustment_args. Duplicate entries (same values at the same time) are deleted.
    """

    # Get lists of all entries for each field in new_field_names
    all_values = retrieve_all_data(
        fields, files_list, time_stamp, filter_dict, df_adjustment_args
    )
    # Get summary stats for this data type and append to all_rows
    all_rows = get_row(fields, all_values)

    columns = [
        "Field",
        "Total",
        "Values",
        "Mean",
        "Min",
        "P1",
        "LQT",
        "median",
        "UQT",
        "P99",
        "Max",
    ]
    df = pd.DataFrame(all_rows, columns=columns)

    return df
