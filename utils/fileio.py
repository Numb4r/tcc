import pandas as pd


def load_csv_file(filePath) -> pd.DataFrame | None:
    file = None
    try:
        file = pd.read_csv(filePath)
        return file
    except:
        raise IOError(f"Could not open file '%s' for reading")


def load_files(data_path: str = "./files/miplib2017-selected.csv", results_path: str = "./files/results.csv") -> (pd.DataFrame, pd.DataFrame):
    data = load_csv_file(data_path)
    data = data.drop("STATUS", axis=1)
    data = data.drop("OBJECTIVE", axis=1)
    results = load_csv_file(results_path)
    key_r = results.keys()
    key_d = data.keys()
    merged = pd.merge(data, results, left_on='NAME',
                      right_on='instance', how='inner')
    data = merged[list(key_d)]
    results = merged[list(key_r)]
    data = data.drop("NAME", axis=1)
    results = results.drop("instance", axis=1)
    return data, results
