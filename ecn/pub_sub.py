import collections
import contextlib
import uuid
from typing import Any, Callable, Generic, Optional, TypeVar

T = TypeVar("T")
S = TypeVar("S")


class Topic(Generic[T]):
    def __init__(self):
        self._subscriptions = collections.OrderedDict()
        self._finished = False

    @property
    def stopped(self):
        return self._finished

    def listen(
        self, callback: Callable[[T], Any], on_finish: Optional[Callable] = None
    ) -> "Subscription":
        if self._finished:
            raise RuntimeError("Cannot listen to topic: stopped")
        uid = uuid.uuid4()
        while uid in self._subscriptions:
            uid = uuid.uuid4()
        sub = Subscription(self, callback, uid, on_finish)
        self._subscriptions[uid] = sub
        return sub

    def _cancel(self, subscription: "Subscription"):
        assert subscription.topic is self
        del self._subscriptions[subscription.uid]
        subscription.on_cancel()

    def _add(self, topic_event: T):
        for sub in self._subscriptions.values():
            sub._callback(topic_event)

    def _finish(self):
        if not self._finished:
            self._finished = True
            subs = tuple(self._subscriptions.values())
            for s in subs:
                s.on_finish()


class Publisher(Generic[T]):
    def __init__(self):
        self._topic = Topic[T]()

    @property
    def topic(self) -> Topic[T]:
        return self._topic

    def add(self, topic_event: T):
        self._topic._add(topic_event)


class Subscription(object):
    def __init__(
        self,
        topic: Topic[T],
        callback: Callable[[T], Any],
        uid: uuid.UUID,
        on_finish: Optional[Callable[[], Any]] = None,
        on_cancel: Optional[Callable[[], Any]] = None,
    ):
        self._topic = topic
        self._uid = uid
        self._done = topic.stopped
        self._callback = callback
        self._on_cancel = on_cancel
        self._on_finish = on_finish

    @property
    def topic(self):
        return self._topic

    @property
    def uid(self):
        return self._uid

    @property
    def done(self):
        return self._done

    def cancel(self):
        if not self._done:
            self._topic._cancel(self)
        else:
            raise RuntimeError("Already done")

    def on_cancel(self):
        if self._done:
            raise RuntimeError("Already done")
        if self._on_cancel is not None:
            self._on_cancel()
        self.on_finish()

    def on_finish(self):
        if not self._done:
            self._done = True
            if self._on_finish is not None:
                self._on_finish()


@contextlib.contextmanager
def accumulator(topic):
    out = []
    sub = topic.listen(out.append)
    yield out
    sub.cancel()


if __name__ == "__main__":
    pub = Publisher()
    with accumulator(pub.topic) as accumulated:
        pub.add("before registering any")
        a = pub.topic.listen(lambda x: print("from a: {}".format(x)))
        pub.add("before registering b")
        b = pub.topic.listen(lambda x: print("from b: {}".format(x)))
        pub.add("after registering b")
        a.cancel()
        pub.add("after cancelling a")
        b.cancel()
        pub.add("after cancelling b")
    print(accumulated)
