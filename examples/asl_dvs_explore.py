from ecn.problems import sources
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# source = sources.asl_dvs_source()
# source = sources.ncaltech101_source()
source = sources.ncars_source()
split = 'train'
total = source.examples_per_epoch(split)
num_events = np.empty((total,), dtype=np.int64)
for i, (example,
        label) in enumerate(tqdm(source.get_dataset(split), total=total)):
    num_events[i] = example['time'].shape[0]

print(np.count_nonzero(num_events > 100000))
print(np.count_nonzero(num_events > 200000))
print(np.count_nonzero(num_events > 300000))
print(np.count_nonzero(num_events > 350000))
print(np.count_nonzero(num_events > 400000))
plt.hist(num_events)
plt.show()

# train counts
# >1: 2424
# >2: 1881
# >3:  504
# >4:  153

# validation counts
# >1: 605
# >2: 472
# >3: 117
# >4:  32

# for example in tqdm(source.get_dataset('train'), total=total):
#     n = example[0]['time'].shape[0]
#     if n > 500000:
#         print(n)
#         sources.vis_example(example, flip_up_down=True)
