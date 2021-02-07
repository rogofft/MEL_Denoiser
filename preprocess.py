import os
import numpy as np
from tqdm import tqdm
from torch import from_numpy, flip, cat


def get_filelist(path) -> list:
    """ Gets list of files from directories which one level belows the 'path' directory.
        :return list of filepaths """
    retlist = []
    # Get subdirectories name list
    dirlist = os.listdir(path)
    # Get list of files from subdirectories
    for directory in dirlist:
        templist = os.listdir(os.path.join(path, directory))
        retlist += list(map(lambda x: os.path.join(path, directory, x), templist))
    return retlist


def get_sampled_data(dirpath, n_loads=None) -> list:
    """ Function loads mel-frequency spectrograms samples [80, 80],
        :n_loads - quantity of files to load from 'dirpath' directory
        :return list of samples """

    # Load n_loads files, or load all
    if n_loads:
        files = get_filelist(dirpath)[:n_loads]
    else:
        files = get_filelist(dirpath)
    # List for return
    samples = []

    # Used for progressbar
    total = len(files)
    loop = tqdm(enumerate(files), total=total, leave=True)

    # This string just for beauty looking in console
    str_data_info = os.path.split(os.path.split(dirpath)[0])[1] + '\\' + os.path.split(dirpath)[1]

    for num, path in loop:
        # Load sample from file
        #samples.append(np.load(path).astype(np.single))
        mel_data = np.load(path).astype(np.single)
        samples += get_window_samples(mel_data, 80)
        
        # Turn progressbar
        loop.set_description(f'Loading {str_data_info}')
    return samples


def get_sample(mel_data, idx, window):
    mirrored_rows = 0
    if idx < window:
        mel_data = cat([flip(mel_data[:, :, 1:window-idx], dims=[2]), mel_data], dim=2)
        mirrored_rows = window - idx
    sample = mel_data[:, :, idx - window + mirrored_rows:idx + 1 + mirrored_rows]
    return sample


def get_window_samples(mel_data, window):
    data_shape = mel_data.shape
    if mel_data.shape[0] % window:
        to_mirror = window - mel_data.shape[0] % window
        mel_data = np.concatenate((mel_data, np.flip(mel_data[-to_mirror-1:-1], axis=0)), axis=0)
    return list(mel_data.reshape(-1, window, data_shape[1]))


def get_raw_data(dirpath, n_loads=None) -> list:
    """ Function loads mel-frequency spectrograms [N, 80],
        :n_loads - quantity of files to load from 'dirpath' directory
        :return list of samples """

    # Load n_loads files, or load all
    if n_loads:
        files = get_filelist(dirpath)[:n_loads]
    else:
        files = get_filelist(dirpath)
    # List for return
    samples = []

    # Used for progressbar
    total = len(files)
    loop = tqdm(enumerate(files), total=total, leave=True)

    # This string just for beauty looking in console
    str_data_info = os.path.split(os.path.split(dirpath)[0])[1] + '\\' + os.path.split(dirpath)[1]

    for num, path in loop:
        # Load data from file
        samples.append(np.load(path).astype(np.single))

        # Turn progressbar
        loop.set_description(f'Loading {str_data_info}')
    return samples
