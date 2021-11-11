import sys
import os


def set_path():
    recursive_unix_dir_backtrack('android_ar')


def recursive_unix_dir_backtrack(desired_dir):
    dir_name = os.getcwd().split('/')[-1]
    if dir_name != desired_dir:
        os.chdir('..')
        recursive_unix_dir_backtrack(desired_dir)
