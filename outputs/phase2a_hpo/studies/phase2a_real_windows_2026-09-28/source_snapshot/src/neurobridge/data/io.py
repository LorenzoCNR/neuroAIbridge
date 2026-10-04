# riuso vecchie fuznioni 
from pathlib import Path
import json
import pickle
import pandas as pd
from scipy.io import loadmat

from pathlib import Path
import json
import pickle
import pandas as pd
from scipy.io import loadmat

def load_data(
    input_dir,
    name,
    file_format,
    sheet_name=0,
    header=0,
    skiprows=None,
    decimal=None,
    thousands=None,
    **read_kwargs
):
    """
    Generic loader for different file formats.

    Args:
        input_dir (str or Path): path to data folder.
        name (str): filename without extension.
        file_format (str): extension: json, pkl, mat, csv, txt, xlsx, xls.
        sheet_name: Excel sheet name/index. 0 = first sheet, None = all sheets.
        header: row to use as column names for csv/excel.
        skiprows: rows to skip for csv/excel.
        decimal: decimal separator for csv/excel, e.g. ",".
        thousands: thousands separator for csv/excel, e.g. ".".
        **read_kwargs: extra pandas read_csv/read_excel arguments.

    Returns:
        object or None.
    """

    input_dir = Path(input_dir).resolve()
    file_format = file_format.lower().replace(".", "")
    path = input_dir / f"{name}.{file_format}"

    print(f"Trying to load: {path}")

    try:
        if file_format == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

        elif file_format == "pkl":
            with open(path, "rb") as f:
                return pickle.load(f)

        elif file_format == "mat":
            return loadmat(path)

        elif file_format in ["xlsx", "xls"]:
            excel_kwargs = {
                "sheet_name": sheet_name,
                "header": header,
                "skiprows": skiprows,
                **read_kwargs
            }

            if decimal is not None:
                excel_kwargs["decimal"] = decimal

            if thousands is not None:
                excel_kwargs["thousands"] = thousands

            data = pd.read_excel(path, **excel_kwargs)

            if isinstance(data, dict):
                print(f"Loaded Excel sheets: {list(data.keys())}")
            else:
                print(f"Loaded Excel sheet='{sheet_name}'")
                print(f"Shape: {data.shape}")

            return data

        elif file_format == "csv":
            encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
            separators = [",", ";"]

            last_error = None

            for enc in encodings:
                for sep in separators:
                    try:
                        csv_kwargs = {
                            "encoding": enc,
                            "sep": sep,
                            "header": header,
                            "skiprows": skiprows,
                            **read_kwargs
                        }

                        if decimal is not None:
                            csv_kwargs["decimal"] = decimal

                        if thousands is not None:
                            csv_kwargs["thousands"] = thousands

                        data = pd.read_csv(path, **csv_kwargs)

                        print(f"Loaded CSV with encoding='{enc}', sep='{sep}'")
                        print(f"Shape: {data.shape}")

                        return data

                    except Exception as e:
                        last_error = e

            print(f"Could not load CSV. Last error: {last_error}")
            return None

        elif file_format == "txt":
            with open(path, "r", encoding="utf-8") as f:
                return f.readlines()

        else:
            print(f"Unsupported format: '{file_format}'")
            return None

    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None

    except Exception as e:
        print(f"Error while loading '{path}': {e}")
        return None

import numpy as np
def preprocess_subject(
    data,
    X_key,
    y_key,
    trial_key,
    trial_length_full,
    steady_length
):
    import numpy as np
    from preprocess import create_trials_id


    X_full = np.asarray(data[X_key])
    y_full = np.asarray(data[y_key]).flatten().astype(int)
    trial_id_full = np.asarray(data[trial_key]).flatten().astype(int)

    n_trials = len(y_full) // trial_length_full

    keep_mask_move   = np.ones(len(y_full), dtype=bool)
    keep_mask_steady = np.zeros(len(y_full), dtype=bool)

    c_t_list_full, trial_len, n_trials = create_trials_id(trial_id_full, y_full)

    for i in range(n_trials):
        start, _ = c_t_list_full[i]
        s_steady = start
        e_steady = start + steady_length
        keep_mask_move[s_steady:e_steady]   = False
        keep_mask_steady[s_steady:e_steady] = True

    X_move       = X_full[keep_mask_move]
    y_move       = y_full[keep_mask_move]
    trial_id_move = trial_id_full[keep_mask_move]

    return X_move, y_move, trial_id_move, c_t_list_full, n_trials