import warnings


# Deprecated import braingeneers.utils.messaging is allowed for backwards compatibility.
# This code should be removed in the future. This was added 27apr2022 by David Parks.
def __getattr__(name):
    if name == 'messaging':
        warnings.warn(
            message='utils.messaging has been deprecated, please import iot.messaging.',
            category=DeprecationWarning,
        )
        from utils import messaging
        return messaging

    elif name == 'NumpyS3Memmap':
        warnings.warn(
            message='utils.NumpyS3Memmap has been deprecated, '
                    'please import utils.numpy_s3_memmap.NumpyS3Memmap.',
            category=DeprecationWarning,
        )
        from utils.numpy_s3_memmap import NumpyS3Memmap
        return NumpyS3Memmap

    else:
        raise AttributeError(name)
