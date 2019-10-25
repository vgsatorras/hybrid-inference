import os
import sys

def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def init_folders(exp_name):
    make_directory('logs')
    make_directory('logs/%s' % exp_name)

def write_file(file, string):
    text_file = open(file, "w")
    text_file.write(string)
    text_file.close()

def redirect_stdout(file):
    sys.stdout = open(file, 'w')

def copy_file(file_name, path):
    os.system("cp %s %s" % (file_name, path))
