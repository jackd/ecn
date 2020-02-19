from .marks import BuiltModels
from .core import batch
from .core import model_input
from .core import assert_mark
from .core import get_mark
from .core import set_mark
from .debug import debug_builder
from .multi_graph import build_multi_graph
from .multi_graph import subgraph
from .multi_model import build_multi_model

__all__ = [
    'batch',
    'model_input',
    'debug_builder',
    'build_multi_graph',
    'subgraph',
    'build_multi_model',
    'BuiltModels',
    'assert_mark',
    'get_mark',
    'set_mark',
]

# import ecn.meta.debug as debug
# import ecn.meta.graph as graph
# import ecn.meta.model as model

# __all__ = [
#     'core',
#     'debug',
#     'graph',
#     'model',
# ]
