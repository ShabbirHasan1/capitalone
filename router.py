"""Path generator to get full path of different directory.
It makes the project portable across different machines.
"""
import os
import sys

from pyutils import mkdir_p

# folder of the project
PROJECT_NAME = 'capitalone'


def fix_root(project_name: str = PROJECT_NAME) -> str:
    """fix_root(project_name)

    Get full path upto PROJECT_NAME. It fixes path issues across different systems.
    :param project_name: folder name of the project
    :return: full path to project folder
    """
    root = os.getcwd()
    first, last = os.path.split(root)
    if last == project_name:
        return root
    else:
        return first


# fix root path in case not running as module
project_root = fix_root()

# other directory path
data = os.path.join(project_root, 'data')  # all data
tmp = os.path.join(data, 'tmp')
fig = os.path.join(data, 'fig')

# other file path
dataset_large = os.path.join(data, 'transactions.csv')
dataset_small = os.path.join(data, 'transactions_small.csv')

# creating different directory if they don't exist
for folder in [data, tmp, fig]:
    mkdir_p(folder)
