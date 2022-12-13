#!/usr/bin/env python-sirius

import random as _random
import numpy as _np
import matplotlib.pyplot as plt

from imaids.models import AppleII as _AppleII
from imaids.models import AppleIISabia as _AppleIISabia
from imaids.blocks import Block as _Block

import idanalysis
#idanalysis.FOLDER_BASE = '/home/ximenes/repos-dev/'
idanalysis.FOLDER_BASE = '/home/gabriel/repos-dev/'

from idanalysis.fmap import EPUOnAxisFieldMap as _EPUOnAxisFieldMap
from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS
from utils import get_idconfig

from run_rk_traj import PHASES, GAPS

class RadiaModelCalibration:
    """."""

    def __init__(self, fmap : _EPUOnAxisFieldMap, epu : _AppleII):
        """."""
        self._fmap = fmap
        self._epu = epu
        self._rz_meas = None
        self._bx_meas = None
        self._by_meas = None
        self._bz_meas = None
        self._rz_model = None
        self._bx_model = None
        self._by_model = None
        self._bz_model = None
        self._nrselblocks = 1

    @property
    def rz_model(self):
        """Longitudinal points where fitting is performed."""
        return self._rz_model

    @property
    def bx_model(self):
        """Model horizontal field."""
        return self._bx_model

    @property
    def by_model(self):
        """Model vertical field."""
        return self._by_model

    @property
    def bz_model(self):
        """Model longitudinal field."""
        return self._bz_model

    @property
    def rz_meas(self):
        """Longitudinal points where field is measured."""
        return self._rz_meas

    @property
    def bx_meas(self):
        """Measured on-axis horizontal field."""
        return self._bx_meas

    @property
    def by_meas(self):
        """Measured on-axis vertical field."""
        return self._by_meas

    @property
    def bz_meas(self):
        """Measured on-axis longitudinal field."""
        return self._bz_meas

    @rz_model.setter
    def rz_model(self, value):
        """Longitudinal points where fitting is performed."""
        self._rz_model = value

    @bx_model.setter
    def bx_model(self, value):
        """Model horizontal field."""
        self._bx_model = value

    @by_model.setter
    def by_model(self, value):
        """Model vertical field."""
        self._by_model = value

    @bz_model.setter
    def bz_model(self, value):
        """Model longitudinal field."""
        self._bz_model = value

    @rz_meas.setter
    def rz_meas(self, value):
        """Longitudinal points where field is measured."""
        self._rz_meas = value

    @bx_meas.setter
    def bx_meas(self, value):
        """Measured on-axis horizontal field."""
        self._bx_meas = value

    @by_meas.setter
    def by_meas(self, value):
        """Measured on-axis vertical field."""
        self._by_meas = value

    @bz_meas.setter
    def bz_meas(self, value):
        """Measured on-axis longitudinal field."""
        self._bz_meas = value

    def set_rz_model(self, nr_pts_period=9):
        """"""
        epu = self._epu
        field_length = 1.1 * (2 + epu.period_length) * epu.nr_periods + 2 * 5 * epu.gap
        z_step = self._epu.period_length / (nr_pts_period - 1)
        maxind = int(field_length / z_step)
        inds = _np.arange(-maxind, maxind + 1)
        rz = inds * z_step
        self._rz_model = rz
        self._by_model = None  # reset field model

    def init_fields(self, nr_pts_period=9):
        """Initialize model and measurement field arrays"""
        self.set_rz_model(nr_pts_period=nr_pts_period)
        field = self._epu.get_field(0, 0, self.rz_model)  # On-axis
        self.bz_model = field[:, 2]
        self.by_model = field[:, 1]
        self.bx_model = field[:, 0]
        fmap = self._fmap
        self.rz_meas = fmap.rz
        self.bx_meas = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
        self.by_meas = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
        self.bz_meas = fmap.bz[fmap.ry_zero][fmap.rx_zero][:]

    def shiftscale_calc_residue(self, shift):
        """Calculate correction scale and residue for a given shift in data."""
        if self.by_model is None and self.bx_model is None:
            raise ValueError('Field model not yet initialized!')
        by_meas_fit = _np.interp(
            self.rz_model, self.rz_meas + shift, self.by_meas)
        bx_meas_fit = _np.interp(
            self.rz_model, self.rz_meas + shift, self.bx_meas)
        bf1 = _np.concatenate((by_meas_fit, bx_meas_fit))
        bf2 = _np.concatenate((self.by_model, self.bx_model))
        scale = _np.dot(bf1, bf2) / _np.dot(bf2, bf2)
        residue = _np.sum((bf1 - scale * bf2)**2)/len(self.rz_model)
        return residue, scale, by_meas_fit, bx_meas_fit

    def shiftscale_plot_fields(self, shift):
        results = self.shiftscale_calc_residue(shift)
        residue, scale, by_meas_fit, bx_meas_fit = results
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(self.rz_model, by_meas_fit, label='meas.')
        axs[0].plot(self.rz_model, scale * self.by_model, label='model')
        axs[1].plot(self.rz_model, bx_meas_fit, label='meas.')
        axs[1].plot(self.rz_model, scale * self.bx_model, label='model')
        sfmt = 'shift: {:.4f} mm, scale: {:.4f}, residue: {:.4f} T'
        fig.suptitle(sfmt.format(shift, scale, residue))
        axs[0].set(ylabel='By [T]')
        axs[1].set(xlabel='rz [mm]', ylabel='Bx [T]')
        plt.xlim(-1600, 1600)
        plt.legend()
        plt.show()
        return by_meas_fit

    def plot_fields(self):
        fig, axs = plt.subplots(2, sharex=True)
        axs[0].plot(self.rz_meas, self.by_meas, label='meas.')
        axs[0].plot(self.rz_model, self.by_model, label='model')
        axs[1].plot(self.rz_meas, self.bx_meas, label='meas.')
        axs[1].plot(self.rz_model, self.bx_model, label='model')
        axs[0].set(ylabel='By [T]')
        axs[1].set(xlabel='rz [mm]', ylabel='Bx [T]')
        plt.xlim(-1600, 1600)
        plt.legend()
        plt.show()

    def shiftscale_set(self, scale):
        """Incorporate fitted scale as effective remanent magnetization."""
        for cas in self._epu.cassettes_ref.values():
            mags_old = _np.array(cas.magnetization_list)
            mags_new = (scale*mags_old).tolist()
            cas.create_radia_object(magnetization_list=mags_new)
        mag_dict = self._epu.magnetization_dict
        self._epu.create_radia_object(magnetization_dict=mag_dict)

    def get_blocks_indices(self):
        nrblocks = self._epu.cassettes_ref['csd'].nr_blocks
        inds_csd = _np.arange(nrblocks)
        inds_cse = _np.arange(nrblocks)
        inds_cie = _np.arange(nrblocks)
        inds_cid = _np.arange(nrblocks)
        _random.shuffle(inds_csd)
        _random.shuffle(inds_cse)
        _random.shuffle(inds_cie)
        _random.shuffle(inds_cid)
        inds = dict(csd=inds_csd[:self._nrselblocks], cse=inds_cse[:self._nrselblocks], cie=inds_cie[:self._nrselblocks], cid=inds_cid[:self._nrselblocks])
        return inds

    def get_selected_blocks_mags(self, blocks_inds):
        mags = dict()
        for cas, ind in blocks_inds.items():
            mags_ = []
            for idx in ind:
                mags_.append(self._epu.cassettes_ref[cas].blocks[idx].magnetization)
            mags[cas] = mags_
        return mags

    def gen_mags_dif(self, blocks_inds, mag_step=1e-2):
        mags_old_dic = self.get_selected_blocks_mags(blocks_inds=blocks_inds)
        mags_dif_dic = dict()
        mags_dif_inverted = dict()
        for cas, mags_old in mags_old_dic.items():
            mags_delta = []
            mags_delta_inv = []
            for mag_old in mags_old:
                mag_delta = mag_step *2*(_np.random.random(3) - 0.5)
                mag_delta_inv = -1*mag_delta
                mags_delta.append(mag_delta.tolist())
                mags_delta_inv.append(mag_delta_inv.tolist())
            mags_dif_dic[cas] = mags_delta
            mags_dif_inverted[cas] = mags_delta_inv
        return mags_dif_dic, mags_dif_inverted

    def update_model_field2(self, blocks_inds, blocks_mags=None):
        """Update model field with new blocks magnetizations."""
        # Example:
        # blocks_inds = {
        #   'csd': [34, 12, 4, 2],
        #   'cse': [3, 1, 7, 3],
        #   'cie': [4, 2, 41, 8],
        #   'cid': [7, 3, 81, 6],
        # }
        #
        # blocks_mags = {
        #   'csd': [m1, m2, m3, ],
        #   'cse': [m4, m5, m6, ],
        #   'cie': [m7, m8, m9, ],
        #   'cid': [m10, m11, m12],
        # }
        #
        # update: self._by_model
        #
        # algorith:
        #
        # 1. build dict blocks_mags_old
        # 2. build dict blocks_mags_diff = blocks_mags - blocks_mags_old
        # 3. calc field_dif on axis for blocks_mags_diff dict
        # 4. add field_dif to self._by_model

        # builds blocks_mags_diff
        blocks_mags_diff = dict()
        for cas, inds in blocks_inds.items():
            mags = self._epu.cassetes_ref[cas].magnetization_list
            blocks_mags_diff[cas] = _np.asarray(blocks_mags[cas]) - mags[inds]

    def update_model_field(self, block_inds, mags_dif):
        """Update By on-axis with new blocks magnetizations."""

        field_dif = _np.zeros((len(self.rz_model), 3))
        for cas_name in block_inds:
            cas = self._epu.cassettes_ref[cas_name]
            for idx_mag, idx in enumerate(block_inds[cas.name]):
                cas.blocks[idx].magnetization = mags_dif[cas.name][idx_mag]
                # cas.blocks[idx].create_radia_object(magnetization=mags_dif[cas.name][idx_mag])
                field_dif += cas.blocks[idx].get_field(x=0, y=0, z=self.rz_model)

        self._by_model += field_dif[:,1]
        return field_dif[:,1]

    def retrogress_model_field(self, block_inds, mags_dif, field_dif):

        for cas_name in block_inds:
            cas = self._epu.cassettes_ref[cas_name]
            for idx_mag, idx in enumerate(block_inds[cas.name]):
                cas.blocks[idx].magnetization = mags_dif[cas.name][idx_mag]
                # cas.blocks[idx].create_radia_object(magnetization=mags_dif[cas.name][idx_mag])
        self._by_model -= field_dif

    def simulated_annealing(self, initial_residue=None, by_meas_fit=None):

        obj_function_old = initial_residue
        for i in _np.arange(10*1500):

            blocks_inds = self.get_blocks_indices()
            delta_mags, delta_mags_inv = self.gen_mags_dif(blocks_inds=blocks_inds, mag_step=0.2*obj_function_old)
            by_dif = self.update_model_field(block_inds=blocks_inds, mags_dif=delta_mags)
            obj_function = _np.sum((by_meas_fit - self.by_model)**2)
            if obj_function < obj_function_old:
                obj_function_old = obj_function
            else:
                self.retrogress_model_field(block_inds=blocks_inds, mags_dif=delta_mags_inv, field_dif=by_dif)

            if i%25 ==0:
                print('residue:',obj_function_old)
                print('iteraction:',i)

        self.plot_fields(by_meas_fit=by_meas_fit)


def get_fmap(phase, gap):
    """."""
    idconfig = get_idconfig(phase, gap)
    MEAS_FILE = ID_CONFIGS[idconfig]
    _, meas_id = MEAS_FILE.split('ID=')
    meas_id = meas_id.replace('.dat', '')
    idkickmap = IDKickMap()
    fmap_fname = FOLDER_BASE + DATA_PATH + MEAS_FILE
    idkickmap.fmap_fname = fmap_fname
    fmap = idkickmap.fmap_config.fmap

    return fmap


def init_objects(phase, gap):
    """."""
    fmap = get_fmap(phase, gap)
    gap = float(gap)
    nr_periods = 54
    period_length = 50
    block_shape = [[[0.1, 0], [40, 0], [40, -40], [0.1, -40]]]
    longitudinal_distance = 0.2
    block_len = period_length/4 - longitudinal_distance
    start_lengths = [block_len/4, block_len/2, 3*block_len/4, block_len]
    start_distances = [block_len/2, block_len/4, 0, longitudinal_distance]
    end_lenghts = start_lengths[-2::-1] # Tirar último elemento e inverter
    end_distances = start_distances[-2::-1] # Tirar último elemento e inverter
    epu = _AppleIISabia(
        gap=gap, nr_periods=nr_periods, period_length=period_length,
        mr=1.25, block_shape=block_shape, block_subdivision=[[1, 1, 1]])#,
        # start_blocks_length=start_lengths, start_blocks_distance=start_distances,
        # end_blocks_length=end_lenghts, end_blocks_distance=end_distances)

    return epu, fmap


def generate_kickmap(posx, posy, phase, gap, radia_model):

    idkickmap = IDKickMap()
    idkickmap.radia_model = radia_model
    idkickmap.beam_energy = 3.0  # [GeV]
    idkickmap.rk_s_step = 2  # [mm]
    idkickmap._radia_model_config.traj_init_px = 0
    idkickmap._radia_model_config.traj_init_py = 0
    idkickmap.traj_init_rz = -1800
    idkickmap.calc_id_termination_kicks(period_len=50, kmap_idlen=2.773)
    print(idkickmap._radia_model_config)
    idkickmap.fmap_calc_kickmap(posx=posx, posy=posy)
    fname = './results/model/kickmap-ID-p{}-g{}.txt'.format(phase, gap)
    idkickmap.save_kickmap_file(kickmap_filename=fname)


def run_calibrated_kickmap(phase, gap):
    """."""
    posx = _np.linspace(-10, +10, 21) / 1000  # [m]
    posy = _np.linspace(-6, +6, 5) / 1000  # [m]
    phase0, gap0 = float(PHASES[2]), GAPS[-2]  # phase 0 mm and gap 32 mm

    print('phase: {}  gap: {}'.format(phase, gap))
    epu, fmap = init_objects(phase=phase, gap=gap)
    cm = RadiaModelCalibration(fmap, epu)
    cm._epu.dp = float(phase)
    cm.init_fields()
    cm.plot_fields()

    # search for best shift and calc scale
    shifts = _np.linspace(-0.25, 0.25, 31) * epu.period_length
    minshift, minscale, minresidue = shifts[0], 1.0, float('inf')
    for shift in shifts:
        residue, scale, *_ = cm.shiftscale_calc_residue(shift=shift)
        # print('shift: {:+08.4f} mm -> residue: {:07.5f} T'.format(
        #     shift, residue))
        if residue < minresidue:
            minshift, minscale, minresidue = shift, scale, residue

    # plot best solution and calibrates model
    by_meas_fit = cm.shiftscale_plot_fields(shift=minshift)
    cm.shiftscale_set(scale=minscale)

    # generate kickmap with calibrated model
    # cm._epu.dg = float(gap) - float(gap0)
    cm.init_fields()
    cm.plot_fields()

    generate_kickmap(
        posx=posx, posy=posy, phase=phase, gap=gap, radia_model=epu)


if __name__ == "__main__":
    phase = PHASES[2]
    gap = '36.0'
    run_calibrated_kickmap(phase, gap)
