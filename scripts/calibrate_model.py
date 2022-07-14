#!/usr/bin/env python-sirius

import random
import numpy as np
from imaids.models import AppleII as _AppleII
from imaids.models import AppleIISabia as _AppleIISabia
from imaids.blocks import Block as _Block
import matplotlib.pyplot as plt
from idanalysis.fmap import FieldmapOnAxisAnalysis as _FieldmapOnAxisAnalysis
from idanalysis.fmap import EPUOnAxisFieldMap as _EPUOnAxisFieldMap


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

    @property
    def rz_model(self):
        """On-axis longitudinal points where fitting is performed."""
        return self._rz_model

    @property
    def field_model(self):
        """Model on-axis vertical field."""
        return self._field_model

    @property
    def rz_meas(self):
        """On-axis longitudinal points where field is measured."""
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
        inds = np.arange(-maxind, maxind + 1)
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
        field_meas_fit = np.interp(
            self.rz_model, self.rz_meas + shift, self.field_meas)
        bf1, bf2 = field_meas_fit, self.field_model
        scale = np.dot(bf1, bf2) / np.dot(bf2, bf2)
        residue = np.sqrt(np.sum((bf1 - scale * bf2)**2)/len(self.rz_model))
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

    def shiftscale_set(self, scale):
        """Incorporate scale into model."""
        raise NotImplementedError

    def update_model_field(self, block_inds, new_mags):
        """Update By on-axis with new blocks magnetizations."""
        # Example:
        # blocks_inds = {
        #   'csd': [34, 12, 4],
        #   'cse': [3, 1, 7],
        #   'cie': [4, 2, 41],
        #   'cid': [7, 3, 81],
        # }
        #
        # mags_new = {
        #   'csd': [m1, m2, m3],
        #   'cse': [m4, m5, m6],
        #   'cie': [m7, m8, m9],
        #   'cid': [m10, m11, m12],
        # }
        #
        # update: self._by_model
        #
        # algorith:
        #
        # 1. build dict mags_old
        # 2. build dict mags_dif = mags_new - mags_old
        # 3. calc field_dif on axis for mags_dif dict
        # 4. add field_dif to self._by_model


def init_objects():

    gap = 22
    nr_periods = 54
    period_length = 50
    block_shape = _Block.PREDEFINED_SHAPES['apple_uvx']
    epu = _AppleIISabia(
        mr=1.24,
        # mr=-0.8769 * 1.24,
        block_shape=block_shape, block_subdivision=[[1, 1, 1]],
        nr_periods=nr_periods, period_length=period_length, gap=gap)
    # print(epu.cassettes_ref['csd'].blocks[0])

    config = _EPUOnAxisFieldMap.CONFIGS.HP_G22P0
    fmap = _EPUOnAxisFieldMap(config=config)
    return epu, fmap


if __name__ == "__main__":
    epu, fmap = init_objects()
    cm = RadiaModelCalibration(fmap, epu)
    cm.init_fields()

    shifts = np.linspace(-0.25, 0.25, 31) * epu.period_length
    # shifts = np.linspace(-7.0, -6.0, 31)
    minshift, minresidue = shifts[0], float('inf')
    for shift in shifts:
        residue, scale, _ = cm.shiftscale_calc_residue(shift=shift)
        print('shift: {:+08.4f} mm -> residue: {:07.5f} T'.format(shift, residue))
        if residue < minresidue:
            minresidue = residue
            minshift = shift

    cm.shiftscale_plot_fields(shift=minshift)

    # cm.shiftscale_set(shift=minshift)
