import numpy as np
import numpy.lib.format as nplf

"""
Helper functions to work with NumPy dtypes.
"""


def dtype_to_descr(dtype: np.dtype):  # list
    return nplf.dtype_to_descr(dtype)


def descr_to_dtype(descr) -> np.dtype:
    # This function is taken verbatim from the source code of NumPy v1.21 since it is not available
    # in some older versions.
    # Source: https://github.com/numpy/numpy/blob/v1.21.0/numpy/lib/format.py#L283-L337
    if isinstance(descr, str):
        # No padding removal needed
        return np.dtype(descr)
    elif isinstance(descr, tuple):
        # subtype, will always have a shape descr[1]
        dt = descr_to_dtype(descr[0])
        return np.dtype((dt, descr[1]))

    titles = []
    names = []
    formats = []
    offsets = []
    offset = 0
    for field in descr:
        if len(field) == 2:
            name, descr_str = field
            dt = descr_to_dtype(descr_str)
        else:
            name, descr_str, shape = field
            dt = np.dtype((descr_to_dtype(descr_str), shape))

        # Ignore padding bytes, which will be void bytes with '' as name
        # Once support for blank names is removed, only "if name == ''" needed)
        is_pad = name == "" and dt.type is np.void and dt.names is None
        if not is_pad:
            title, name = name if isinstance(name, tuple) else (None, name)
            titles.append(title)
            names.append(name)
            formats.append(dt)
            offsets.append(offset)
        offset += dt.itemsize

    return np.dtype(
        {
            "names": names,
            "formats": formats,
            "titles": titles,
            "offsets": offsets,
            "itemsize": offset,
        }
    )
