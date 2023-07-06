import functools
import warnings


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


def report(func):
    @functools.wraps(func)
    def wrapper_report(*args, **kwargs):
        df = kwargs["df"] if "df" in kwargs.keys() else args[0]
        print(f"{df.shape[0]} rows before filter operation.")
        df = func(*args, **kwargs)
        print(f"{df.shape[0]} rows after filter operation.")

        return df

    return wrapper_report
