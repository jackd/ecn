# TODO

* work out which models I used in paper: inception_vox_pooling
  * N-MINST: nmnist-aug-inf-vox-rlrp.gin
  * MNIST-DVS: mnist-dvs-aug8-vox.gin
  * CIFAR10-DVS: cifar10-aug8-vox.gin - batch_size == 16, not 8?
  * N-CALTECH101: ncaltech101-aug8-small-b-vox.gin ? spike threshold == 1.25 not 1.2
  * ASL-DVS: asl-dvs-small-vox.gin
* update pipelines / source
* change to kblocks multigraph
* verify on small examples
* move visualization scripts elsewhere
* remove other configs / models
* cifar10-dvs - fix transpose from dataset change?

## MAYBE

* replace pub / sub implementation with pypubsub
* fixed `bucket_sizes = False`
