import torch


def combine_dims(x: torch.Tensor, dim_begin: int, dim_end: int):
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)


def remove_dim(x: torch.Tensor, dim: int):
    return combine_dims(x, dim - 1, dim + 1)


def pretty_time(seconds):
    hours = seconds // 3600
    minutes = seconds // 60 % 60
    seconds %= 60
    res = ""
    if hours:
        res += '%d h' % hours
        res += ' '
        res += '%d m' % minutes
        res += ' '
        res += '%d s' % int(seconds)
    elif minutes:
        res += '%d m' % minutes
        res += ' '
        res += '%d s' % int(seconds)
    else:
        res += '%.2f s' % seconds
    return res


def inverse_mapping(m: dict) -> dict:
    res = {}
    for e in m:
        res[m[e]] = e
    return res
