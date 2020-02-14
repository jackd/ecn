from typing import Tuple, Optional, MutableMapping
import abc
import tensorflow as tf
from kblocks.tf_typing import TensorLike


class BuiltModels(object):
    PRE_BATCH = 'pre_batch'
    POST_BATCH = 'post_batch'
    TRAINED = 'trained'

    @classmethod
    def validate(cls, id_: str):
        if id_ not in cls.all():
            raise ValueError('invalid PipelineModel key {}'.format(id_))

    @classmethod
    def all(cls):
        return (BuiltModels.PRE_BATCH, BuiltModels.POST_BATCH,
                BuiltModels.TRAINED)


class Marks(MutableMapping[tf.Tensor, str]):

    def __init__(self, base: MutableMapping[tf.Tensor, str]):
        self._base = base

    def __delitem__(self, key):
        del self._base[key]

    def __iter__(self):
        return iter(self._base)

    def __len__(self):
        return len(self._base)

    @abc.abstractmethod
    def _inputs(self, x) -> Tuple[tf.Tensor, ...]:
        raise NotImplementedError

    def __getitem__(self, x: TensorLike) -> Optional[str]:
        return self._propagate(x)

    def __setitem__(self, x: TensorLike, mark: str) -> None:
        # propagate plus check consistency
        m = self._propagate(x)
        if m is not None and m != mark:
            raise ValueError(
                'Attempted to mark x with {}, got inputs with mark {}'.format(
                    mark, m))
        # propagate marks down dependencies
        # self._propagate_down(x, mark)

    def __contains__(self, x: TensorLike) -> bool:
        return self._propagate(x) is None

    # don't think we ever need to propagate down now that I think about it...
    # def _propagate_down(self, x: TensorLike, mark: str) -> None:
    #     if x not in self._base:
    #         self._base[x] = mark
    #         for i in self._inputs(x):
    #             self._propagate_down(i, mark)

    def _propagate(self, end: TensorLike) -> Optional[str]:
        mark = self._base.get(end)
        if mark is not None:
            return mark
        inputs = self._inputs(end)
        if len(inputs) == 0:
            return None
        # get marks from all inputs, ensuring consistency
        for i in inputs:
            mi = self._propagate(i)
            if mi is not None:
                if mark is None:
                    mark = mi
                elif mi != mark:
                    raise ValueError(
                        'different marks detected: {} and {}'.format(mark, mi))
        if mark is None:
            return None
        self._base[end] = mark
        return mark
