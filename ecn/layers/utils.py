import functools
import inspect
from typing import Callable

import tensorflow as tf


def _call(args, _fn, _arg_names, **kwargs):
    kwargs.update(zip(_arg_names, args))
    return _fn(**kwargs)


def _as_layer(fn, is_tensor_fn: Callable):
    argspec = inspect.getfullargspec(fn)
    arg_names = argspec.args

    @functools.wraps(fn)
    def ret_fn(*args, **kwargs):
        assert not any(n in kwargs for n in arg_names[: len(args)])
        kwargs.update(zip(arg_names, args))
        args = []
        names = []
        for name in arg_names:
            if name in kwargs:
                value = kwargs[name]
                if is_tensor_fn(value):
                    args.append(value)
                    names.append(name)
        for name in names:
            del kwargs[name]

        kwargs.update(dict(_fn=fn, _arg_names=names))
        if len(args) == 0:
            return _call(args, **kwargs)

        return tf.keras.layers.Lambda(_call, arguments=kwargs)(args)

    return ret_fn


if tf.version.VERSION < "2.4":

    def _is_symbolic_tensor(tensor):
        if isinstance(tensor, tf.Tensor):
            return hasattr(tensor, "graph")
        if isinstance(tensor, (tf.RaggedTensor, tf.SparseTensor)):
            component_tensors = tf.nest.flatten(tensor, expand_composites=True)
            return any(hasattr(t, "graph") for t in component_tensors)
        if isinstance(tensor, tf.Variable):
            return (
                getattr(tensor, "_keras_history", False) or not tf.executing_eagerly()
            )
        return False

    as_layer = functools.partial(_as_layer, is_tensor_fn=_is_symbolic_tensor)
else:
    as_layer = functools.partial(_as_layer, is_tensor_fn=tf.is_tensor)
