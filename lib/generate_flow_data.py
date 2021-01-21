import numpy as np
import pandas as pd
import argparse

def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)

def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data[line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        print(mean, std)
        yield x, y

def generate_data(graph_signal_matrix_filename, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    data = np.expand_dims(data, -1)
    # keys = data.keys()
    # if 'train' in keys and 'val' in keys and 'test' in keys:
    #     for i in generate_from_train_val_test(data, transformer):
    #         yield i
    # elif 'data' in keys:
    length = data.shape[0]
    for i in generate_from_data(data, length, transformer):
        yield i
    # else:
        # raise KeyError("neither data nor train, val, test is in the data")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="data/", help="Output directory."
    )
    parser.add_argument(
        "--traffic_npy_filename",
        type=str,
        default="data/pems5flownew.npy",
        help="Raw traffic readings.",
    )
    args = parser.parse_args()

    graph_signal_matrix_filename = args.traffic_npy_filename
    name = ['train', 'val', 'test']
    for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
        print(x.shape, y.shape)
        np.savez(args.output_dir+f'{name[idx]}.npz', x=x, y=y)
        
