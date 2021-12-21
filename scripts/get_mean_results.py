
import json

def main():

    path = '../output_bad_conditions_anfi_radisson/aesthetic/'

    methods = [
    'DW_C_WB_output.json',
    'G_CC_WB_output.json',
    'GT_output.json',
    'SCB_PILSATUR_output.json'
    ]

    for method_file in methods:
        with open(path + method_file) as f:
            data = json.load(f)
            data = data['results']

            mean = 0

            for img in data:
                mean += img['mean_score_prediction']

            print(method_file.split('.')[0] + '\t : mean : ', str(mean / len(data)))



    print('done')



if __name__ == '__main__':
    main()