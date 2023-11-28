import datasets
import os

dset = datasets.load_dataset("imagenet-1k")
dset.save_to_disk(f'{os.environ["DATASET_STORE"]}/prepared_hf/imagenet1k.hfdatasets')
dset.cleanup_cache_files()