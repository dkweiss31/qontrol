from __future__ import annotations

import glob
import os
import re

import h5py


def generate_file_path(extension: str, file_name: str, path: str) -> str:
    # Ensure the path exists.
    os.makedirs(path, exist_ok=True)  # noqa: PTH103

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for file_name_ in glob.glob(os.path.join(path, '*')):  # noqa: PTH118, PTH207
        if f'_{file_name}.{extension}' in file_name_:
            numeric_prefix = int(
                re.match(r'(\d+)_', os.path.basename(file_name_)).group(1)  # noqa: PTH119
            )
            max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)

    # Generate the file path.
    return os.path.join(  # noqa: PTH118
        path, f'{str(max_numeric_prefix + 1).zfill(5)}_{file_name}.{extension}'
    )


def extract_info_from_h5(filepath: str) -> [dict, dict]:
    data_dict = {}
    with h5py.File(filepath, 'r') as f:
        for key in f:
            data_dict[key] = f[key][()]
        param_dict = dict(f.attrs.items())
    return data_dict, param_dict


def append_to_h5(filepath: str, data_dict: dict, param_dict: dict) -> None:
    with h5py.File(filepath, 'a') as f:
        for key, val in data_dict.items():
            if val.shape[0] == 0: # takes care of an edge case
                continue
            if key not in f:
                f.create_dataset(
                    key, data=val, chunks=True, maxshape=(None, *val.shape[1:])
                )
            else:
                f[key].resize(f[key].shape[0] + val.shape[0], axis=0)
                f[key][-val.shape[0]:] = val
        for key, val in param_dict.items():
            try:
                f.attrs[key] = val
            except TypeError:
                f.attrs[key] = str(val)
