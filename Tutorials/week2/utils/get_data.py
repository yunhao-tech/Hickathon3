import os

import pandas as pd
from tqdm import notebook


FILES_FOLDER = {
    'raw': ['raw/building_metadata.feather', 'raw/meters.feather', 'raw/weather.feather'],
    'clean': ['clean/building_metadata.feather', 'clean/meters.feather', 'clean/weather.feather'],
    'merged': ['merged/data.feather'],
    'model': ['model/train.feather', 'model/val.feather', 'model/test.feather']
}


def load_data(folder, data_dir, dict_files=FILES_FOLDER):
    """Load data files from a folder `folder` in a directory `data_dir`
    """
    files = dict_files[folder]

    dataframes = []

    print(":: Start loading data")
    for name_file in notebook.tqdm(files):
        dataframe = pd.read_feather(os.path.join(data_dir, name_file))
        dataframes.append(dataframe)

    return dataframes
