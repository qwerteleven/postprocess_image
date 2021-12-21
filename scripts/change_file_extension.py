import glob
import os

from shutil import copyfile

def main(path):
    
    files = glob.glob(path + '/*.png', recursive=True)

    for file in files:

        new_file = path + '/' + file.split('\\')[-1][:-4] + '.jpeg'
        copyfile(file, new_file)

        os.remove(file)
        
        print(new_file)



if __name__ == '__main__' :
    path = '../postprocess_image - 1/postprocess_image/deep-white-balance'
    main(path)