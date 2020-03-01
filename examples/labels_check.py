import numpy as np
from tqdm import tqdm
from ecn.problems import sources

# source = sources.ncaltech101_source()
source = sources.ncars_source()

dataset = source.get_dataset('train')
total = source.examples_per_epoch('train')
label_set = set()

for example, label in tqdm(dataset, total=total):
    label_set.add(label.numpy())

print(sorted(label_set))
print(len(label_set))
# print(np.all(np.arange(102) == np.array(sorted(label_set))))
# turns out caltech 101 has 102 labels...
