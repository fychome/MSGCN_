import argparse
import numpy as np
import os
import sys
import yaml

from lib.utils import load_graph_data
from model.msgcn_supervisor import MSGCNSupervisor


def run_msgcn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        adj_mx, mask = load_graph_data(graph_pkl_filename)

        supervisor = MSGCNSupervisor(adj_mx=adj_mx, mask=mask, **supervisor_config)
        mean_score, outputs = supervisor.evaluate('test')
        np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='config/msgcn_metrla.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='data/msgcn_predictions_metrla.npz')
    args = parser.parse_args()
    run_msgcn(args)
