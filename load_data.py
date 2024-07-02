import os
import torch
import shutil
import numbers
import requests
import warnings
import numpy as np
from importlib import import_module

def load_data(name, cache_dir=None):
    """
    Data loading function. See `data repository
    <https://github.com/pygod-team/data>`_ for supported datasets.
    For injected/generated datasets, the labels meanings are as follows.

    - 0: inlier
    - 1: contextual outlier only
    - 2: structural outlier only
    - 3: both contextual outlier and structural outlier

    Parameters
    ----------
    name : str
        The name of the dataset.
    cache_dir : str, optional
        The directory for dataset caching.
        Default: ``None``.

    Returns
    -------
    data : torch_geometric.data.Data
        The outlier dataset.

    Examples
    --------
    >>> from pygod.utils import load_data
    >>> data = load_data(name='weibo') # in PyG format
    >>> y = data.y.bool()    # binary labels (inlier/outlier)
    >>> yc = data.y >> 0 & 1 # contextual outliers
    >>> ys = data.y >> 1 & 1 # structural outliers
    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.pygod/data')
    file_path = os.path.join(cache_dir, name+'.pt')
    zip_path = os.path.join(cache_dir, name+'.pt.zip')

    if os.path.exists(file_path):
        data = torch.load(file_path)
    else:
        url = "https://github.com/pygod-team/data/raw/main/" + name + ".pt.zip"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        shutil.unpack_archive(zip_path, cache_dir)
        data = torch.load(file_path)
    return data