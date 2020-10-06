# Event-stream Convolutional Network

[tensorflow](https://github.com/tensorflow/tensorflow) implementation of event stream networks from [Sparse Convolutions on Continuous Domains](https://github.com/sccd).

## Quick-start

```bash
pip install tensorflow==2.3  # 2.2 also works
git clone https://github.com/jackd/ecn.git
pip install -e ecn
# train nmnist model
python -m ecn '$KB_CONFIG/fit' '$ECN_CONFIG/nmnist-aug-inf-vox-rlrp.gin'
```
