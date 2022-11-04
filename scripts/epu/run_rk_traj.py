#!/usr/bin/env python-sirius

from scipy import optimize as optimize

from mathphys.functions import save_pickle, load_pickle
from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS

_phase25n = ['ID4083', 'ID4103', 'ID4088', 'ID4093', 'ID4098']  # phase -25
_phase16n = ['ID4081', 'ID4101', 'ID4086', 'ID4091', 'ID4096']  # phase -16
_phase00p = ['ID4079', 'ID4099', 'ID4084', 'ID4089', 'ID4094']  # phase  00
_phase16p = ['ID4080', 'ID4100', 'ID4085', 'ID4090', 'ID4095']  # phase  16
_phase25p = ['ID4082', 'ID4102', 'ID4087', 'ID4092', 'ID4097']  # phase  25

DEF_RK_S_STEP = 0.2  # [mm]
GAPS = ['22.0', '23.3', '25.7', '29.3', '40.9']
PHASES = ['-25.00', '-16.39', '+00.00', '+16.39', '+25.00']
CONFIGS = [_phase25n, _phase16n, _phase00p, _phase16p, _phase25p]


def create_idkickmap(idconfig):
    """."""
    # get fieldmap file name
    MEAS_FILE = ID_CONFIGS[idconfig]
    _, meas_id = MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE

    # create  IDKickMap and set beam energy, fieldmap filename and RK step
    idkickmap = IDKickMap()
    idkickmap.fmap_fname = fmap_fname
    idkickmap.beam_energy = 3.0  # [GeV]
    # # print(idkickmap.brho)

    # set various fmap_configurations
    idkickmap.fmap_config.traj_init_rz = 1 * min(idkickmap.fmap.rz)
    idkickmap.fmap_config.traj_rk_min_rz = 1 * max(idkickmap.fmap.rz)

    return idkickmap


def calc_rk_traj(configs, traj_init_rx, traj_init_ry, rk_s_step=DEF_RK_S_STEP):
    """Calculate RK for set of EPU configurations."""
    bx, by, bz = dict(), dict(), dict()
    rz, rx, ry = dict(), dict(), dict()
    px, py = dict(), dict()
    i1bx, i2bx = dict(), dict()
    i1by, i2by = dict(), dict()
    fmapbx, fmapby, fmaprz = dict(), dict(), dict()
    
    fieldtools = FieldmapOnAxisAnalysis()
    for gap, idconfig in zip(GAPS, configs):
        print('gap: {} mm'.format(gap))
        # create IDKickMap and calc trajectory
        idkickmap = create_idkickmap(idconfig)
        idkickmap.rk_s_step = rk_s_step
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)
        traj = idkickmap.traj
        fmap = idkickmap.fmap_config.fmap

        fmaprz[gap] = fmap.rz
        fmapbx[gap] = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
        fmapby[gap] = fmap.by[fmap.ry_zero][fmap.rx_zero][:]

        bx[gap], by[gap], bz[gap] = traj.bx, traj.by, traj.bz
        rx[gap], ry[gap], rz[gap] = traj.rx, traj.ry, traj.rz
        px[gap], py[gap] = traj.px, traj.py

        i1bx_ = fieldtools.calc_first_integral(traj.bx, traj.rz)
        i1by_ = fieldtools.calc_first_integral(traj.by, traj.rz)
        i1bx[gap], i1by[gap] = i1bx_, i1by_
        i2bx[gap] = fieldtools.calc_second_integral(i1bx_, traj.rz)
        i2by[gap] = fieldtools.calc_second_integral(i1by_, traj.rz)

    data = dict()
    data['bx'], data['by'], data['bz'] = bx, by, bz
    data['rx'], data['ry'], data['rz'] = rx, ry, rz
    data['px'], data['py'] = px, py
    data['fmapbx'], data['fmapby'] = fmapbx, fmapby
    data['fmaprz'] = fmaprz
    data['i1bx'], data['i1by'] = i1bx, i1by
    data['i2bx'], data['i2by'] = i2bx, i2by
    return data


def save_rk_traj(rk_s_step=DEF_RK_S_STEP):
    """"""
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]

    data = dict()
    for phase, configs in zip(PHASES, CONFIGS):
        print('phase {} mm, configs: {}'.format(phase, configs))
        data_ = \
            calc_rk_traj(configs, traj_init_rx, traj_init_ry, rk_s_step)
        data[phase] = data_
        print()

    rk_traj_data = dict()
    rk_traj_data['traj_init_rx'] = traj_init_rx
    rk_traj_data['traj_init_ry'] = traj_init_ry
    rk_traj_data['rk_s_step'] = rk_s_step
    rk_traj_data['data'] = data
    fpath = './results/phase-organized/'
    save_pickle(rk_traj_data, fpath + 'rk_traj_data.pickle', overwrite=True)
    

def load_rk_traj():
    fpath = './results/phase-organized/'
    traj_data = load_pickle(fpath + 'rk_traj_data.pickle')
    traj_init_rx = traj_data['traj_init_rx']
    traj_init_ry = traj_data['traj_init_ry']
    rk_s_step = traj_data['rk_s_step']
    data = traj_data['data']
    return data, traj_init_rx, traj_init_ry, rk_s_step


if __name__ == "__main__":
    """."""
    rk_s_step = DEF_RK_S_STEP
    save_rk_traj(rk_s_step)
