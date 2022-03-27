#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt

from idanalysis.trajectory import IDTrajectory

from pyaccel.optics.edwards_teng import estimate_coupling_parameters
from idanalysis.epudata import EPUData
from idanalysis.model import calc_optics, create_model, get_id_sabia_list

from utils import create_epudata


def print_configs(configs):

    data = dict()
    for config in configs:
        label = configs.get_config_label(config)
        header = configs.get_header(config)
        print(label, header)


data = r = create_epudata()
print_configs(data)

