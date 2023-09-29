import inspect
import sys
from typing import Union

try:
    # noinspection PyProtectedMember
    from pip._internal.operations import freeze
except ImportError:  # pip < 10.0
    # noinspection PyUnresolvedReferences
    from pip.operations import freeze


def set_default_kwargs(keyword_dict: Union[dict, None], default_dict: dict):
    """
    Compares a default parameter dict with the user-provided and updates the latter if necessary.

    Parameters
    ----------
    keyword_dict: dict or None
        user-supplied kwargs dict
    default_dict: dict
        Standard settings that should be applied if not specified differently by the user.
    """
    if keyword_dict is None:
        return default_dict
    for k, v in default_dict.items():
        if k not in keyword_dict.keys():
            keyword_dict[k] = v

    return keyword_dict


def generate_environment_txt():
    pkgs = freeze.freeze()

    with open("environment.txt", 'w') as env_:
        env_.write('\n'.join(pkgs))
