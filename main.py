import os
import sys
import pandas as pd
import time
import argparse

import router
import dataset
import model_factory


def gen_small_dataset():
    dataset.gen_small_dataset(small_dataset_path=router.dataset_small, large_dataset_path=router.dataset_large)


def convert_dataset_json_to_csv():
    st_time = time.time()
    path_from = os.path.join(router.project_root, 'transactions.txt')
    dataset.dataframe_json_to_csv(path_from=path_from, path_to=router.dataset_large)
    print(f'Time needed: {time.time()-st_time:.2f} seconds')


def run_all_models():
    model_factory.run_all_models()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=['json_to_csv', 'gen_small', 'explore_data'],
                        required=True, help="Which program to run?")
    parser.add_argument("--what_data", choices=['small', 'large'], required=False, help="Which dataset to use?")
    args = parser.parse_args()

    # track time needed to run different module
    st_time = time.time()

    if args.which == 'json_to_csv':
        convert_dataset_json_to_csv()
    elif args.which == 'gen_small':
        gen_small_dataset()
    elif args.which == 'explore_data':
        if args.what_data == 'small':
            dset = dataset.DatasetC1(name='small', path=router.dataset_small)
        elif args.what_data == 'large':
            dset = dataset.DatasetC1(name='large', path=router.dataset_large)
        else:
            raise ValueError('Invalid arguments of what_data')
        dset.describe()
    else:
        raise ValueError('Invalid option given')

    print(f'Total processing time: {time.time()-st_time:.2f} sec', file=sys.stderr)


if __name__ == '__main__':
    main()
