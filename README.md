# Event-stream Convolutional Network

[tensorflow](https://github.com/tensorflow/tensorflow) implementation of event stream networks from [Sparse Convolutions on Continuous Domains](https://github.com/sccd).

## Quick-start

```bash
pip install tensorflow==2.3  # 2.2, tf-nightly also works
git clone https://github.com/jackd/ecn.git
pip install -e ecn
# train nmnist model
python -m ecn '$KB_CONFIG/trainables/fit' '$ECN_CONFIG/sccd/nmnist.gin'
```

## Visualize Data

You can preprocess the leaky-integrate-and-fire streams with

```bash
python -m ecn '$ECN_CONFIG/vis/streams2d.gin' '$ECN_CONFIG/sccd/nmnist.gin'
```

or the event neighborhoods with

```bash
python -m ecn '$ECN_CONFIG/vis/adj.gin' '$ECN_CONFIG/sccd/nmnist.gin'
```

## Saved Data

Running with the default configurations will result in data files created in:

- `~/tensorflow_datasets/`: downloads, extracted files and basic preprocessing of events into `tfrecords` files.
- `~/ecn/`: cached datasets (potentially hundreds of GBs) and model checkpoints / training summaries.

## Known Issues

- `pytest ecn` sometimes results in a test failure in `ecn/ops/conv_test.py:test_csr_gradient`. This does not occur with `python ecn/ops/conv_test.py`.
- `killed` (memory leak): some cache implementations seem to cause memory issues. The exact source is unknown, but `kb.data.snapshot` and `kb.data.save_load_cache` are likely culprits. Try `kb.data.tfrecords_cache` (though creating the cache is significantly slower).
