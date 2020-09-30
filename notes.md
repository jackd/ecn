# TODO

* verify on small examples
* move visualization scripts elsewhere
* remove problems dir
* remove other configs / models

## MAYBE

* resolve: slow down over first epochs for `nmnist-aug-inf-vox-rlrp.gin`
* replace pub / sub implementation with pypubsub
* fixed `bucket_sizes = False`

## DONE

* work out which models I used in paper: inception_vox_pooling
  * N-MINST: nmnist-aug-inf-vox-rlrp.gin
  * MNIST-DVS: mnist-dvs-aug8-vox-rlrp.gin
  * CIFAR10-DVS: cifar10-aug8-vox.gin - batch_size == 16, not 8?
  * N-CALTECH101: ncaltech101-aug8-small-b-vox.gin ? spike threshold == 1.25 not 1.2
  * ASL-DVS: asl-dvs-small-vox.gin
* update pipelines / source
* change to kblocks multigraph
* resolve: integration tests hang
* cifar10-dvs - fix transpose from dataset change?
