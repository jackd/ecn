from typing import Optional, Union, TypeVar
import abc
from typing import List
import tensorflow as tf
from kblocks.tf_typing import TensorLike, TensorLikeSpec

T = TypeVar('T')


class MetaBuilderContext(object):
    _stack: List['MetaBuilderContext'] = []

    def __init__(self, batch_size: Optional[int] = None):
        self._batch_size = batch_size

    @staticmethod
    def get_default() -> 'MetaBuilderContext':
        stack = MetaBuilderContext._stack
        if len(stack) == 0:
            raise RuntimeError('MetaBuilderContext stack empty')
        return stack[-1]

    @staticmethod
    def has_default() -> bool:
        return len(MetaBuilderContext._stack) > 0

    def __enter__(self: T) -> T:
        MetaBuilderContext._stack.append(self)
        return self

    def __exit__(self, type, value, traceback):
        top = MetaBuilderContext._stack.pop()
        assert (top is self)

    @property
    def batch_size(self) -> Optional[int]:
        return self._batch_size

    @abc.abstractmethod
    def batch(self, tensor: TensorLike):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def model_input(self, tensor: TensorLike, name: Optional[str] = None):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def get_mark(self, x: TensorLike):
        raise NotImplementedError('Abstract method')

    @abc.abstractmethod
    def set_mark(self, x: TensorLike, mark: str):
        raise NotImplementedError('Abstract method')

    # @abc.abstractmethod
    # def learning_phase(self):
    #     raise NotImplementedError('Abstract method')


get_default = MetaBuilderContext.get_default
has_default = MetaBuilderContext.has_default


def get_batch_size() -> Optional[int]:
    return get_default().batch_size


def batch(tensor: TensorLike) -> TensorLike:
    return get_default().batch(tensor)


def model_input(tensor: TensorLike, name: Optional[str] = None) -> TensorLike:
    return get_default().model_input(tensor, name=name)


def get_mark(x: TensorLike):
    return get_default().get_mark(x)


def set_mark(x: TensorLike, mark: str):
    return get_default().set_mark(x, mark)


def assert_mark(x: TensorLike, mark: str, name: str = 'tensor'):
    actual = get_mark(x)
    if actual != mark:
        raise ValueError('Expected {} to have mark {}, but has {}'.format(
            name, mark, actual))


# def learning_phase() -> tf.Tensor:
#     if has_default():
#         return get_default().learning_phase()
#     else:
#         return tf.keras.backend.learning_phase()
