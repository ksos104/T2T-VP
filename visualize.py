import os
import numpy as np
import matplotlib.pyplot as plt


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def make_array():
    from PIL import Image
    return np.array([np.asarray(Image.open('face.png').convert('RGB'))]*12)


def main():
    dst_path = 'vis_result'
    os.makedirs(dst_path, exist_ok=True)

    root_path = 'results_T2T/Debug/results/Debug/sv'
    input_list = np.load(os.path.join(root_path, 'inputs.npy'))
    trues_list = np.load(os.path.join(root_path, 'trues.npy'))
    preds_list = np.load(os.path.join(root_path, 'preds.npy'))

    print("input_data.shape: ", input_list.shape)
    print("trues_data.shape: ", trues_list.shape)
    print("preds_data.shape: ", preds_list.shape)

    num_batch = input_list.shape[0]

    for idx in range(num_batch):
        input_batch = input_list[idx]
        true_batch = trues_list[idx]
        pred_batch = preds_list[idx]

        num_sample = input_batch.shape[0]

        fig = plt.figure(figsize=(8, 8))
        column = num_sample
        row = 3

        for i in range(3):
            for j in range(num_sample):
                fig.add_subplot(row, column, i*10+j+1)
                if i == 0:
                    plt.imshow(input_batch[j].transpose(1,2,0))
                elif i == 1:
                    plt.imshow(true_batch[j].transpose(1,2,0))
                elif i == 2:
                    plt.imshow(pred_batch[j].transpose(1,2,0))
        
        plt.savefig('vis_result/{}.png'.format(idx))


if __name__ == '__main__':
    main()