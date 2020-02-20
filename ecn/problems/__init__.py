import os
from kblocks.gin_utils.config import try_register_config_dir

ECN_CONFIG_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), 'configs'))

try_register_config_dir('ECN_CONFIG', ECN_CONFIG_DIR)

from . import builders
from . import sources

__all__ = [
    'builders',
    'sources',
    'ECN_CONFIG_DIR',
]
