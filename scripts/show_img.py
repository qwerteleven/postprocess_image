import sys
import json
import matplotlib.pyplot as plt
import cv2

def main(img_index):


    path_metadata_img = '../test_image_aesthetics/'
    img_data = get_img_ordered(path_metadata_img)
    path = '../test_image_aesthetics/'


    f, axarr = plt.subplots(1, 2)


    axarr[0].imshow(cv2.imread(path + 'before/' + img_data[0][img_index][2] + '.jpg')[:, :, ::-1])
    axarr[0].set_title('before:  ' + img_data[0][img_index][2] + '.jpg' + '  score: ' + str(img_data[0][img_index][1] ))
    
    axarr[1].imshow(cv2.imread(path + 'after/' + img_data[0][img_index][2] + '.jpg')[:, :, ::-1])
    axarr[1].set_title('after:  ' + img_data[1][img_index][2] + '.jpg' + '  score: ' + str(img_data[1][img_index][1] ))


    plt.show()



    print('done')



def get_img_ordered(path):

    methods = [
    'aesth_before_output.json',
    'aesth_after_output.json'
    ]

    results = []
    for method_file in methods:
        with open(path + method_file) as f:
            data = json.load(f)

            data = data['results']

            output = []


            for img in data:
                output.append((img['image_id'].split('_')[0], img['mean_score_prediction'], img['image_id']))

            results.append(sorted(output,  key=lambda x: x[0]))

    return results


if __name__ == '__main__':

    img_index = int(sys.argv[1])

    main(img_index)