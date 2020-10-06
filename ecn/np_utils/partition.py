import numba as nb
import numpy as np

IntArray = np.ndarray


@nb.njit()
def temporal_partition(event_times: IntArray, stop_times: IntArray) -> IntArray:
    """
    Args:
        event_times: [in_size] int array of input sizes
        stop_times: [out_size] int array or partition end times

    Returns:
        [out_size] int array of indices into event_times of index splits.
    """
    num_events = event_times.size
    indices = np.empty((stop_times.size,), dtype=np.int64)
    i = 0

    for stop_index in range(stop_times.size):
        stop_time = stop_times[stop_index]
        while i < num_events and event_times[i] <= stop_time:
            i += 1
        indices[stop_index] = i
        if i == num_events:
            indices[stop_index:] = num_events
            break
    else:
        raise RuntimeError()

    return indices
