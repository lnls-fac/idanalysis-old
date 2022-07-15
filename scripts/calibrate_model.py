#!/usr/bin/env python-sirius

import numpy as _np
import random as _random
from imaids.models import AppleII as _AppleII
from imaids.models import AppleIISabia as _AppleIISabia
from imaids.blocks import Block as _Block
import matplotlib.pyplot as plt
from idanalysis.fmap import EPUOnAxisFieldMap as _EPUOnAxisFieldMap
from copy import deepcopy

class RadiaModelCalibration:
    """."""

    def __init__(self, fmap : _EPUOnAxisFieldMap, epu : _AppleII):
        """."""
        self._fmap = fmap
        self._epu = epu
        self._rz_meas = None
        self._field_meas = None
        self._rz_model = None
        self._field_model = None
        self._nrselblocks = 5

    @property
    def rz_model(self):
        """Longitudinal points where fitting is performed."""
        return self._rz_model

    @property
    def field_model(self):
        """Model vertical field."""
        return self._field_model

    @property
    def rz_meas(self):
        """Longitudinal points where field is measured."""
        return self._rz_meas
    
    @property
    def field_meas(self):
        """Measured on-axis vertical field."""
        return self._field_meas

    def set_rz_model(self, nr_pts_period=9):
        epu = self._epu
        field_length = 1.1 * (2 + epu.period_length) * epu.nr_periods + 2 * 5 * epu.gap
        z_step = self._epu.period_length / (nr_pts_period - 1)
        maxind = int(field_length / z_step)
        inds = _np.arange(-maxind, maxind + 1)
        rz = inds * z_step
        self._rz_model = rz
        self._field_model = None  # reset field model

    def init_fields(self, nr_pts_period=9):
        """Initialize model and measurement field arrays"""
        self.set_rz_model(nr_pts_period=nr_pts_period)
        field = self._epu.get_field(0, 0, self.rz_model)  # On-axis
        self._field_model = field[:, 1]
        fmap = self._fmap
        self._rz_meas = fmap.rz
        self._field_meas = fmap.by[fmap.ry_zero, fmap.rx_zero, :]
        
    def shiftscale_calc_residue(self, shift):
        """Calculate correction scale and residue for a given shift in data."""
        if self.field_model is None:
            raise ValueError('Field model not yet initialized!')
        field_meas_fit = _np.interp(
            self.rz_model, self.rz_meas + shift, self.field_meas)
        bf1, bf2 = field_meas_fit, self.field_model
        scale = _np.dot(bf1, bf2) / _np.dot(bf2, bf2)
        residue = _np.sum((bf1 - scale * bf2)**2)/len(self.rz_model)
        return residue, scale, field_meas_fit

    def shiftscale_plot_fields(self, shift):
        residue, scale, field_meas_fit = self.shiftscale_calc_residue(shift)
        plt.plot(self.rz_model, field_meas_fit, label='meas.')
        plt.plot(self.rz_model, scale * self.field_model, label='model')
        sfmt = 'shift: {:.4f} mm, scale: {:.4f}, residue: {:.4f} T'
        plt.title(sfmt.format(shift, scale, residue))
        plt.xlim(-1600, 1600)
        plt.xlabel('rz [mm]')
        plt.ylabel('By [T]')
        plt.legend()
        plt.show()

    def plot_fields(self):
        plt.plot(self.rz_model, self.field_model, label='model')
        plt.xlim(-1600, 1600)
        plt.xlabel('rz [mm]')
        plt.ylabel('By [T]')
        plt.legend()
        plt.show()    

    def shiftscale_set(self, scale):
        """Incorporate fitted scale as effective remanent magnetization."""
        for cas in self._epu.cassettes_ref.values():
            mags_old = _np.array(cas.magnetization_list)
            mags_new = (scale*mags_old).tolist()
            cas.create_radia_object(magnetization_list=mags_new)

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

    #def gen_new_mags(self, nrselblocks=5):


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
        # update: self._field_model
        #
        # algorith:
        #
        # 1. build dict blocks_mags_old
        # 2. build dict blocks_mags_diff = blocks_mags - blocks_mags_old
        # 3. calc field_dif on axis for blocks_mags_diff dict
        # 4. add field_dif to self._field_model

        # builds blocks_mags_diff
        blocks_mags_diff = dict()
        for cas, inds in blocks_inds.items():
            mags = self._epu.cassetes_ref[cas].magnetization_list
            blocks_mags_diff[cas] = _np.asarray(blocks_mags[cas]) - mags[inds]

    def update_model_field(self, block_inds, new_mags):
        """Update By on-axis with new blocks magnetizations."""
        mags_old = self._epu.magnetization_dict
        mags_dif = dict()
        field_dif = _np.zeros((len(self.rz_model), 3))
        for cas_name in block_inds:
            cas = self._epu.cassettes_ref[cas_name]
            mags_dif_list = []
            for idx_mag, idx in enumerate(block_inds[cas.name]):
                mags_dif_list.append((_np.array(new_mags[cas.name][idx_mag])-_np.array(mags_old[cas.name][idx])).tolist())
                
            mags_dif[cas.name] = mags_dif_list
            for idx_mag, idx in enumerate(block_inds[cas.name]):
                cas.blocks[idx].create_radia_object(magnetization=mags_dif[cas.name][idx_mag])
                field_dif += cas.blocks[idx].get_field(x=0, y=0, z=self.rz_model)
       
        self._field_model += field_dif[:,1]
        
        
def init_objects(phase, gap):
    """."""
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
        mr=1.25, block_shape=block_shape, block_subdivision=[[1, 1, 1]],
        start_blocks_length=start_lengths, start_blocks_distance=start_distances,
        end_blocks_length=end_lenghts, end_blocks_distance=end_distances)
    # print(epu.cassettes_ref['csd'].blocks[0])

    configs = {
        (0, 22.0) : _EPUOnAxisFieldMap.CONFIGS.HP_G22P0,
        (0, 25.7) : _EPUOnAxisFieldMap.CONFIGS.HP_G25P7,
    }
    try:
        config = configs[(phase, gap)]
    except KeyError:
        raise NotImplementedError
    fmap = _EPUOnAxisFieldMap(config=config)
    return epu, fmap


if __name__ == "__main__":

    # create objects and init fields
    phase = 0
    # gap = 22  # [mm]
    gap = 25.7  # [mm]
    epu, fmap = init_objects(phase=0, gap=gap)
    cm = RadiaModelCalibration(fmap, epu)
    # cm.update_model_field(blocks_inds={'csd': [0, ]})
    # raise ValueError
    cm.init_fields()

    # search for best shift and calc scale
    shifts = _np.linspace(-0.25, 0.25, 31) * epu.period_length
    minshift, minscale, minresidue = shifts[0], 1.0, float('inf')
    for shift in shifts:
        residue, scale, _ = cm.shiftscale_calc_residue(shift=shift)
        print('shift: {:+08.4f} mm -> residue: {:07.5f} T'.format(shift, residue))
        if residue < minresidue:
            minshift, minscale, minresidue = shift, scale, residue

    # plot best solution and calibrates model
    cm.shiftscale_plot_fields(shift=minshift)
    cm.shiftscale_set(scale=minscale)
    
    m = 1.0001
    #Example:

    
    mags_new = {
      'csd': [(m*_np.array(epu.cassettes['csd'].blocks[0].magnetization)).tolist()],
    #   'cse': [(m*_np.array(epu.cassettes['cse'].blocks[3].magnetization)).tolist(), (m*_np.array(epu.cassettes['cse'].blocks[1].magnetization)).tolist(), (m*_np.array(epu.cassettes['cse'].blocks[7].magnetization)).tolist()],
    #   'cie': [(m*_np.array(epu.cassettes['cie'].blocks[4].magnetization)).tolist(), (m*_np.array(epu.cassettes['cie'].blocks[2].magnetization)).tolist(), (m*_np.array(epu.cassettes['cie'].blocks[41].magnetization)).tolist()],
    #   'cid': [(m*_np.array(epu.cassettes['cid'].blocks[7].magnetization)).tolist(), (m*_np.array(epu.cassettes['cid'].blocks[3].magnetization)).tolist(), (m*_np.array(epu.cassettes['cid'].blocks[81].magnetization)).tolist()],
    }
    
    blocks_inds = cm.get_blocks_indices()
    print(blocks_inds)
    #cm.update_model_field(block_inds=blocks_inds, new_mags=mags_new)
    cm.plot_fields()