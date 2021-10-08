
import glob
import os
from datetime import datetime

from shutil import copyfile


def main(path):

    
    files = glob.glob(path + '/**/*.jpeg', recursive=True)

    for file in files:
        copyfile(file, path + '\\' + str(datetime.now()).replace(':', '-') + '.jpeg')

    print('done')

if __name__ == '__main__':
    path = os.getcwd() + '\\images'
    main(path)