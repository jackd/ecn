import gin
import numpy as np
from events_tfds.events import asl_dvs, cifar10_dvs, mnist_dvs, ncaltech101, nmnist

gin.constant("ASL_DVS_GRID_SHAPE", asl_dvs.GRID_SHAPE)
gin.constant("CIFAR10_DVS_GRID_SHAPE", cifar10_dvs.GRID_SHAPE)
gin.constant("MNIST_DVS_GRID_SHAPE", mnist_dvs.GRID_SHAPE)
gin.constant("NCALTECH101_GRID_SHAPE", ncaltech101.GRID_SHAPE)
gin.constant("NMNIST_GRID_SHAPE", nmnist.GRID_SHAPE)

gin.constant("ASL_DVS_NUM_CLASSES", asl_dvs.NUM_CLASSES)
gin.constant("CIFAR10_DVS_NUM_CLASSES", cifar10_dvs.NUM_CLASSES)
gin.constant("MNIST_DVS_NUM_CLASSES", mnist_dvs.NUM_CLASSES)
gin.constant("NCALTECH101_NUM_CLASSES", ncaltech101.NUM_CLASSES)
gin.constant("NMNIST_NUM_CLASSES", nmnist.NUM_CLASSES)

gin.constant("PI_ON_8", np.pi / 8)
gin.constant("NEG_PI_ON_8", -np.pi / 8)
