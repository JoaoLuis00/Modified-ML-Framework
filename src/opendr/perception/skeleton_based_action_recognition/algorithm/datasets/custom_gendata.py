import argparse
import pickle
from tqdm import tqdm
import numpy as np
import os
import math
import pandas
from pathlib import Path

CUSTOM_CLASSES = pandas.read_csv(Path(__file__).parent / 'custom_labels.csv', verbose=True, index_col=0).to_dict()["name"]