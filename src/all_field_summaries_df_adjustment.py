import numpy as np


def df_adjustment(df, df_adjustment_args):
    """
    Adjusts the incoming dataframe df, based on the values in the list df_adjustments_args.
    """

    if df_adjustment_args[0] == "steps":
        _dict_ = df_adjustment_args[1]
        df = df.sort_values(by=_dict_["timestamp"]).reset_index(drop=True)
        if _dict_["cumulative"]:
            df["new steps"] = df[_dict_["steps col"]].diff().fillna(0)
            if _dict_["device col"] != None:
                df["same_device"] = (
                    df[[_dict_["device col"]]] == df[[_dict_["device col"]]].shift()
                ).any(axis=1)
                df = df[df["same_device"] == True]
            df[_dict_["steps col"]] = df["new steps"]
            df = df[df[_dict_["steps col"]] >= 0]
        if _dict_["steps per second"]:
            df["gap"] = df[_dict_["timestamp"]].diff().fillna(0)
            df["sps"] = df[_dict_["steps col"]] / df["gap"]
            df["sps"] = np.where(df["gap"] != 0, df[_dict_["steps col"]] / df["gap"], 0)
            df[_dict_["steps col"]] = df["sps"]
        if _dict_["delete zeros"]:
            df = df[df[_dict_["steps col"]] > 0]

    return df
