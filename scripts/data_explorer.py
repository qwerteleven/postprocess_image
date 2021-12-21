
import json
import numpy as np
import matplotlib.pyplot as plt

def get_diff(B, A):

    result = []

    for i in range(len(A)):
        for j in range(len(B)):
            if A[i][0] == B[j][0]:
                result.append(round(A[i][1] - B[j][1], 3))
                break

    return result



def main():

    # path = '../MLSP_anfi_radisson/aesthetic/'

    path = '../test_image_aesthetics/'

    methods = [
    'aesth_before_output.json',
    'aesth_after_output.json',

    ]

    results = []

    for method_file in methods:
        with open(path + method_file) as f:
            data = json.load(f)

            data = data['results']

            output = []


            for img in data:
                output.append((img['image_id'].split('_')[0], img['mean_score_prediction'], img['image_id']))

            results.append(sorted(output,  key=lambda x: x[2]))
            # results.append(output)

    print('list images: ', [(index, i[0]) for index, i in enumerate(results[0])])
            

    # in order to modify the size
    fig, axs = plt.subplots(1, 2)


    result = get_diff(results[0], results[0])
    axs[0].plot(result)
    axs[0].set_ylim([-1, 1])
    axs[0].grid(color='r', linestyle='-', linewidth=1)
    axs[0].plot([0, 25], [0, 0], 'b-')
    min_value = min(result)
    min_index = result.index(min_value)
    print(results[0][min_index][2])

    axs[0].set_title('before')
    axs[0].set_xticks(range(0, 25, 5))

    result = get_diff(results[0], results[1])
    axs[1].plot(result, 'tab:orange')
    min_value = min(result)
    min_index = result.index(min_value)
    print(results[1][min_index][2])
    axs[1].set_title('after')
    axs[1].set_ylim([-1, 1])
    axs[1].plot([0, 25], [0, 0], 'b-')
    axs[1].grid(color='r', linestyle='-', linewidth=1)
    axs[1].set_xticks(range(0, 25, 5))



    plt.show()





    # in order to modify the size
    fig, axs = plt.subplots(2, 2)


    result = [round(results[0][i][1], 0) for i in range(len(results[0]))]
    axs[0, 0].hist(result)
    axs[0, 0].set_ylim([0, 25])
    axs[0, 0].grid(color='r', linestyle='-')
    axs[0, 0].set_title('before')
    axs[0, 0].set_xlim([3, 7])

    result = [round(results[1][i][1], 0) for i in range(len(results[1]))]
    axs[0, 1].hist(result)
    axs[0, 1].set_ylim([0, 25])
    axs[0, 1].grid(color='r', linestyle='-')
    axs[0, 1].set_title('after')
    axs[0, 1].set_xlim([3, 7])


    plt.show()


    print('done')



if __name__ == '__main__':
    main()