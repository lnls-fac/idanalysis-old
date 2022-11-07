#!/usr/bin/env python-sirius

import numpy as np

from idanalysis import IDKickMap

from utils import ID_PERIOD, ID_KMAP_LEN

from run_rk_traj import DEF_RK_S_STEP
from run_rk_traj import create_idkickmap
from run_rk_traj import create_kmap_filename
from run_rk_traj import CONFIGS, PHASES, GAPS


def calc_kmap(phase, gap, traj_init_px, traj_init_py, posx, posy, rk_s_step):
    """."""
    idkickmap = create_idkickmap(phase, gap)
    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_config.traj_init_px = traj_init_px
    idkickmap.fmap_config.traj_init_py = traj_init_py
    print(idkickmap.fmap_config)
    idkickmap.calc_id_termination_kicks(
        period_len=ID_PERIOD, kmap_idlen=ID_KMAP_LEN, plot_flag=False)
    print(idkickmap.fmap_config)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = create_kmap_filename(phase, gap)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


if __name__ == "__main__":
    """."""
    traj_init_px = 0  # [rad]
    traj_init_py = 0  # [rad]
    posx = np.linspace(-18, +18, 37) / 1000  # [m]
    posy = np.linspace(-12, +12, 3) / 1000  # [m]
    for phase in PHASES:
        for gap in GAPS:
            calc_kmap(
                phase, gap,
                traj_init_px, traj_init_py,
                posx, posy,
                rk_s_step=DEF_RK_S_STEP)
