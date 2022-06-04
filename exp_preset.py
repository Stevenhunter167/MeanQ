import copy
import code
import traceback
import tqdm
import pprint
import argparse
from pprint import pprint
from trainers import *

from execution_plans import basic_experiment
from configs import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--game', type=str, required=True)
    parser.add_argument('--preset', type=str, required=True)
    parser.add_argument('--verbose', type=int, default=0)
    args = argparse.Namespace(**vars(parser.parse_args()), **eval(parser.parse_args().preset))
    basic_experiment(args)