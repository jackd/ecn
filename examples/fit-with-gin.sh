# demonstrate custom configuration
python -m ecn \
    '$KB_CONFIG/trainables/fit' \
    '$ECN_CONFIG/sccd/nmnist.gin' \
    '$ECN_CONFIG/data/aug/offline.gin' \
    --bindings='
variant_id = "f64"  # affects save directories, but not cache file locations
# we cann change the number of filters without affecting the cache
# if we change something that affects pre_cache operations
# (e.g. add a new convolution grid)
# we should change `family_id` instead / as well.
filters0 = 64
cache_repeats = 4
cache_factory = @kb.data.snapshot
'
