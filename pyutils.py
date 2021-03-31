import os
import sys

import pandas as pd
import matplotlib.pyplot as plt


def mkdir_p(dir, verbose=False, backup_existing=False, if_contains=None):
    '''make a directory (dir) if it doesn't exist'''
    # todo: add recursive directory creation
    # todo: give a msg when dir exist but empty
    # dir: given full path only create last name as directory
    # if_contains: backup_existing True if_contains is a list. If there is any file/directory in the
    #              first (not recursive) pass match with if_contains then backup
    if not os.path.exists(dir):  # directory does not exist
        if verbose is True: print(f'Created new dir named: {dir}', file=sys.stderr)
        os.mkdir(dir)
    else:  # dir exist
        need_backup = False
        # check if backup is needed
        if backup_existing and len(os.listdir(dir)) > 0:
            # check contains helper
            def helper_contains():
                # check if if_contains is a list
                if not isinstance(if_contains, list):
                    raise ValueError('if_contains in mkdir_p is not a list')

                files_inside = os.listdir(dir)
                for pat in if_contains:
                    if any(pat in file for file in files_inside):
                        return True
                return False

            if (if_contains is None) or (if_contains and helper_contains()):
                need_backup = True

        if need_backup:
            # find new path that doesn't exist
            for i in range(10000):
                new_dir_path = f'{dir}_{i}'
                if not os.path.exists(new_dir_path):
                    break
            # renaming directory
            if verbose:
                print(f'Moving dir {dir} -> {new_dir_path}', file=sys.stderr)
            os.rename(src=dir, dst=new_dir_path)
            # now creating dir
            os.mkdir(dir)
    return dir

# ===============================================
# Pandas utility functions
# ===============================================


def pd_set_display(max_col=True, max_row=True, col_wrap=False):
    """
    Set to display all rows and columns in pandas dataframe
    :param max_col:
    :param max_row:
    :param col_wrap: wrap up the line while printing
    :return:
    """
    if max_col:
        pd.set_option("max_columns", None)  # Showing only two columns
    if max_row:
        pd.set_option("max_rows", None)

    if not col_wrap:
        pd.set_option('display.expand_frame_repr', False)


def header(line: str, type_=1):
    """
    Print line within a block for easy visualization
    :param line: line to pring
    :param type_: type of the header to use
    :return:
    """
    if type_ == 1:
        """
        =========================
        This is an example header
        =========================
        """
        eq = "".join(['=' for _ in range(len(line))])
        para = f'{eq}\n{line}\n{eq}'
    elif type_ == 2:
        """
        -------------------------
        This is an example header
        -------------------------
        """
        eq = "".join(['-' for _ in range(len(line))])
        para = f'{eq}\n{line}\n{eq}'
    else:
        raise ValueError('Invalid type_')

    print(para)


def plt_save(path):
    """
    Save plot in the given path
    :param path:
    :return:
    """
    plt.grid()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()