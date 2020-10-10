# Event-stream Convolutional Network

[tensorflow](https://github.com/tensorflow/tensorflow) implementation of event stream networks from [Sparse Convolutions on Continuous Domains](https://github.com/sccd).

## Quick-start

```bash
pip install tensorflow==2.3  # 2.2 also works
git clone https://github.com/jackd/ecn.git
pip install -e ecn
# train nmnist model
python -m ecn '$KB_CONFIG/fit' '$ECN_CONFIG/sccd/nmnist.gin'
```

## Saved Data

Running with the default configurations will result in data files created in:

- `~/tensorflow_datasets/`: downloads, extracted files and basic preprocessing of events into `tfrecords` files.
- `~/ecn/`: cached datasets (potentially hundreds of GBs) and model checkpoints / training summaries.

## Known Issues

- `pytest ecn` sometimes results in a test failure in `ecn/ops/conv_test.py:test_csr_gradient`. This does not occur with `python ecn/ops/conv_test.py`.
