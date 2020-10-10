import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ecn import sources

# from ecn import vis

# source = sources.asl_dvs_source()
# source = sources.ncaltech101_source()
source = sources.ncars_source()
split = "train"
total = source.epoch_length(split)
num_events = np.empty((total,), dtype=np.int64)
for i, (example, label) in enumerate(tqdm(source.get_dataset(split), total=total)):
    num_events[i] = example["time"].shape[0]

# limits = [100000, 200000, 300000, 400000]
limits = [1000, 2000, 4000, 8000, 16000, 32000]

for lim in limits:
    print(lim, np.count_nonzero(num_events > lim))

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
#         vis.vis_example(example, flip_up_down=True)
