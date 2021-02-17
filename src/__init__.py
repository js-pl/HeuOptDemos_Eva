from genericpath import exists
import os
import sys
path = os.getcwd()
sys.path.append(path + os.path.sep + 'pymhlib_fork_copy')
os.makedirs('logs' + os.path.sep + 'saved', exist_ok=True)
os.makedirs('logs' + os.path.sep + 'saved_runtime', exist_ok=True)
