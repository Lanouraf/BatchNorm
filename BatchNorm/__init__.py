from .BatchNorm import BatchNorm


def BatchNorm2D(filters, epsilon=1e-5):
    """
    2D version of BatchNorm block
    :param filters: Number of input filters
    :param epsilon: Constant for numerical stability
    :return: BatchNorm block module
    """
    return BatchNorm(filters, epsilon, dims='2D')


def BatchNorm1D(filters, epsilon=1e-5):
    """
    1D version of BatchNorm block
    :param filters: Number of input filters
    :param epsilon: Constant for numerical stability
    :return: BatchNorm block module
    """
    return BatchNorm(filters, epsilon, dims='1D')
