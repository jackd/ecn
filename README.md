# [Event-stream Convolutional Network](https://github.com/jackd/ecn)

[tensorflow](https://github.com/tensorflow/tensorflow) implementation of event stream networks from _Sparse Convolutions on Continuous Domains_, ACCV2020.

- [ACCV 2020 Paper](paper)
- [sccd repository](https://github.com/jackd/sccd)
- [Spotlight Video](https://youtu.be/OihcDbfT1ks) (1min)
- [Oral Presentation](https://youtu.be/26GDhWfU280) (9min)

```tex
@InProceedings{Jack_2020_ACCV,
    author    = {Jack, Dominic and Maire, Frederic and Denman, Simon and Eriksson, Anders},
    title     = {Sparse Convolutions on Continuous Domains for Point Cloud and Event Stream Networks},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}
```

## Quick-start

### Installation

```bash
pip install tensorflow  # or tf-nightly - must be >=2.3
git clone https://github.com/jackd/ecn.git
cd ecn
pip install -r requirements.txt
pip install -e .
```

### Train NMNIST Model

```bash
python -m ecn '$KB_CONFIG/trainables/fit' '$ECN_CONFIG/sccd/nmnist.gin'
# other results from sccd paper are configured in ecn/configs/sccd/*
tensorboard --logdir=~/ecn/nmnist/sccd/default_variant/run-01/
```

### Visualize Data

You can preprocess the leaky-integrate-and-fire streams with

```bash
python -m ecn '$ECN_CONFIG/vis/streams2d.gin' '$ECN_CONFIG/sccd/nmnist.gin'
```

or the event neighborhoods with

```bash
python -m ecn '$ECN_CONFIG/vis/adj.gin' '$ECN_CONFIG/sccd/nmnist.gin'
```

### `python -m ecn`

`python -m ecn` is a light wrapper around `python -m kblocks` which exposes `$ECN_CONFIG` for command line argument just like `$KB_CONFIG` is exposed in `kblocks`. In particular, note that `$ECN_CONFIG` is set inside the python script, so must be passed as a string, e.g. `python -m ecn '$ECN_CONFIG/foo'` rather than `python -m ecn $ECN_CONFIG/foo`. See the examples subdirectory and [kblocks repository](kblocks) and for more examples.

## Custom Package Dependencies

This project depends on multiple custom python packages. These are:

- [kblocks](kblocks) for experiment management, configuration via [gin-config](https://github.com/google/gin-config) and various tensorflow utilities.
- [meta-model](https://github.com/jackd/meta-model) for simultaneously building and connecting the multiple models associated with data pipelining and model training.
- [tfrng](https://github.com/jackd/tfrng) for random number generation and deterministic, pre-emptible data pipelining.
- [wtftf](https://github.com/jackd/wtftf) for keras layer wrappers for composite tensors in tensorflow 2.3.
- [events-tfds](https://github.com/jackd/events-tfds) for [tensorflow-datasets](https://github.com/tensorflow/datasets) implementations that manage dataset downloading and model-independent preprocessing for event stream datasets.
- [numba-stream](https://github.com/jackd/numba-neighbors) for [numba](https://github.com/numba/numba) implementations of stream subsampling and event neighborhood calculations.

## Saved Data

Running with the default configurations will result in data files created in:

- `~/tensorflow_datasets/`: downloads, extracted files and basic preprocessing of events into `tfrecords` files.
- `~/ecn/`: configuration logs, model checkpoints, training summaries and cached datasets (potentially hundreds of GBs - see below).

## Caching

Implementations for problems apart from NMNIST use offline caching of data to perform event neighborhood computations. This has a number of effects.

- Early epoch will take longer to run than later epochs, depending on the number of `cache_repeats` for the problem.
- Potentially very large files will be created and stored on disk (by default under `~/ecn/`).
- Some cache implementations may cause very slow memory leaks that will eventually lead to a process being killed.

We provide 4 cache implementations in `kblocks.data`:

| `kblocks.data` method  | Based on                                 | Lazy/Eager | Supports Compression | Possible memory leak |
|----------------------- |------------------------------------------|------------|----------------------|----------------------|
| `snapshot` (default)   | `tf.data.experimental.snapshot`          | Lazy       | [x]                  | [x]                  |
| `cache`                | `tf.data.Dataset.cache`                  | Lazy       | []                   | []                   |
| `save_load_cache`      | `tf.data.experimental.[save,load]`       | Eager      | [x]                  | [x]                  |
| `tfrecords_cache`      | `kblocks.data.tfrecords.tfrecords_cache` | Eager      | [x]                  | []                   |

## Known Issues

- `pytest ecn` sometimes results in a test failure in `ecn/ops/conv_test.py:test_csr_gradient`. This does not occur with `python ecn/ops/conv_test.py`.
- `killed` (memory leak): some cache implementations seem to cause memory issues. The exact source is unknown, but `kb.data.snapshot` and `kb.data.save_load_cache` are likely culprits. Try `kb.data.tfrecords_cache` (though creating the cache is significantly slower).

[paper]: https://openaccess.thecvf.com/content/ACCV2020/html/Jack_Sparse_Convolutions_on_Continuous_Domains_for_Point_Cloud_and_Event_ACCV_2020_paper.html
[kblocks]: https://github.com/jackd/kblocks
