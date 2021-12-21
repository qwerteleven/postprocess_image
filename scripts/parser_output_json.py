import json
import os


def main():
    path = '../MLSP_anfi_radisson/'


    files = [
        'DW.txt',
        'G_CC.txt',
        'GT.txt',
        'pilsatur.txt'
    ]

    methods = [
        'DW_C_WB_output.json',
        'G_CC_WB_output.json',
        'GT_output.json',
        'SCB_PILSATUR_output.json'
    ]

    for index, file in enumerate(files):
        f = open(path + file)
        output = []

        for line in f.readlines():
            parts = line.split(' ')
            data_img = {
                "image_id": str(parts[-1].split('.')[0]),
                "mean_score_prediction": float(parts[3])
            }

            output.append(data_img)
        
        results = {"results" : output}

   
        with open(path + 'aesthetic/' + methods[index], 'w') as f:
            json.dump(results, f, indent=4)



    print('Done')





if __name__ == '__main__':
    main()