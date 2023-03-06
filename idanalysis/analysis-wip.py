#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
from idanalysis import IDKickMap as _IDKickMap
from scipy import integrate as _integrate
from mathphys.functions import save_pickle as _save_pickle
from mathphys.functions import load_pickle as _load_pickle


class AnalysisFromRadia:

    def __init__(self, var_params, constant_params, models):
        # Model attributes
        self.rx_max = None  # [mm]
        self.rx_nrpts = None
        self.rz_max = None  # [mm]
        self.rz_nrpts = None
        self.roll_off_rx = None  # [mm]
        self.var_params = None
        self.constant_params = None
        self.models = None
        self.model_params = None
        self.data = dict()

        # Trajectory attributes
        self.idkickmap = None
        self.beam_energy = 3.0  # [Gev]
        self.rk_s_step = 1.0  # [mm]
        self.traj_init_rz = None  # [mm]
        self.traj_max_rz = None  # [mm]
        self.kmap_idlen = None  # [m]

        self.FOLDER_DATA = './results/model/data/'

        self._set_var_params(var_params)
        self._set_constant_params(constant_params)
        self._set_models(models)
        self._set_models_params()

    @property
    def idkickmap(self):
        """Return an object of IDKickMap class.

        Returns:
            IDKickMap object:
        """
        return self.idkickmap

    @property
    def models(self):
        """Return a dictionary with all ID models.

        Returns:
            Dictionary: A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        return self.models

    @property
    def var_params(self):
        """Names of variable parameters of the models.

        Returns:
            var_params (tuple): tuple of strings
                example: ('gap', 'width')
        """
        return self.var_params

    @property
    def constant_params(self):
        """Names of constant parameters of the models.

        Returns:
            constant_params (tuple): tuple of strings
                example: ('period_length', 'phase')
        """
        return self.constant_params

    @property
    def model_params(self):
        """Value of models's constant parameters.

        Returns:
            dictionary: The keys are the names of the parameters
                (output of constant_params property) and the values
                are the values of the paremeters.
        """
        return self.model_params

    @models.setter
    def _set_models(self, models):
        """Set models attribute.

        Args:
            models (dictionary): A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        self.models = models

    @var_params.setter
    def _set_var_params(self, var_params):
        """Define variable parameters of the models.

        Args:
            var_params (tuple): tuple of strings
                example: ('gap', 'width')
        """
        self.var_params = var_params

    @constant_params.setter
    def _set_constant_params(self, constant_params):
        """Define constant parameters of the models.

                These parameters that are the same for
                every model of the model's dictionary.

        Args:
            constant_params (tuple): tuple of strings
                example: ('period_length', 'phase')
        """
        self.constant_params = constant_params

    @model_params.setter
    def _set_models_params(self):
        """Set model's constant parameters."""
        self.model_params = dict()
        key = self.models.keys()
        for param in self.constant_params:
            self.model_params[param] = getattr(self.models[key[0]], param)

    @idkickmap.setter
    def _config_traj(self, traj_init_rz, traj_max_rz, kmap_idlen):
        """Set idkickmap config for trajectory."""
        self.idkickmap = _IDKickMap()
        self.idkickmap.beam_energy = self.beam_energy
        self.idkickmap.rk_s_step = self.rk_s_step
        self.idkickmap.traj_init_rz = traj_init_rz
        self.idkickmap.traj_rk_min_rz = traj_max_rz
        self.idkickmap.kmap_idlen = kmap_idlen
        self.traj_init_rz = traj_init_rz
        self.traj_max_rz = traj_max_rz
        self.kmap_idlen = kmap_idlen

    def _get_field_roll_off(self, rx, peak_idx=0, field_component='by'):
        """Calculate the roll-off of a field component.

        Args:
            rx (numpy 1D array): array with positions where the field will be
                calculated
            peak_idx (int): Peak index where the roll-off will be calculated.
                Defaults to 0.
            field_component (str): Component of the field, bx or by.
                Defaults to 'by'.
        """
        b_x = dict()
        rx_dict = dict()
        roll_off = dict()
        comp_idx = self._get_field_component_idx(field_component)
        for var_params, id in self.models.items():
            b_ = list()
            period = self.model_params['period_length']
            rz = _np.linspace(-period/2, period/2, 201)
            field = id.get_field(0, 0, rz)
            b = field[:, comp_idx]
            b_max_idx = _np.argmax(b)
            rz_at_max = rz[b_max_idx] + peak_idx*period
            field = id.get_field(rx, 0, rz_at_max)
            b = field[:, comp_idx]

            roff_idx = _np.argmin(_np.abs(rx-self.roll_off_rx))
            rx0_idx = _np.argmin(_np.abs(rx))
            roff = _np.abs(b[roff_idx]/b[rx0_idx]-1)

            b_x[var_params] = b
            rx_dict[var_params] = rx
            roll_off[var_params] = roff

        self.data['rolloff_rx'] = rx_dict
        self.data['rolloff_{}'.format(field_component)] = b
        self.data['rolloff_value'] = roll_off

    def _get_field_on_axis(self, rz):
        """Get the field along z axis.

        Args:
            rz (numpy 1D array): array with longitudinal positions where the
                field will be calculated
        """
        bx_dict, by_dict, bz_dict, rz_dict = dict(), dict(), dict(), dict()
        for var_params, id in self.models.items():
            field = id.get_field(0, 0, rz)
            bx_dict[var_params] = field[:, 0]
            by_dict[var_params] = field[:, 1]
            bz_dict[var_params] = field[:, 2]
            rz_dict[var_params] = rz
        self.data['onaxis_bx'] = bx_dict
        self.data['onaxis_by'] = by_dict
        self.data['onaxis_rz'] = rz_dict

    def _get_field_on_trajectory(self):
        bx, by, bz = dict(), dict(), dict()
        rz, rx, ry = dict(), dict(), dict()
        px, py, pz = dict(), dict(), dict()
        s = dict()
        self._config_traj()
        for var_params, id in self.models.items():
            # create IDKickMap and calc trajectory
            self.idkickmap.radia_model = id
            self.idkickmap.fmap_calc_trajectory(
                traj_init_rx=0, traj_init_ry=0,
                traj_init_px=0, traj_init_py=0)
            traj = self.idkickmap.traj
            bx[var_params], by[var_params], bz[var_params] =\
                traj.bx, traj.by, traj.bz
            rx[var_params], ry[var_params], rz[var_params] =\
                traj.rx, traj.ry, traj.rz
            px[var_params], py[var_params], pz[var_params] =\
                traj.px, traj.py, traj.pz
            s[var_params] = traj.s

        self.data['ontraj_bx'] = bx
        self.data['ontraj_by'] = by
        self.data['ontraj_bz'] = bz

        self.data['ontraj_rx'] = rx
        self.data['ontraj_ry'] = ry
        self.data['ontraj_rz'] = rz

        self.data['ontraj_px'] = px
        self.data['ontraj_py'] = py
        self.data['ontraj_pz'] = pz

        self.data['ontraj_s'] = s

    def _save_data(self):
        value = self.data['ontraj_s']
        fname = self.FOLDER_DATA + 'field_data'
        for keys in list(value.keys()):
            fdata = dict()
            for info, value in self.data.items():
                fdata[info] = value[keys]
            for i, parameter in enumerate(keys):
                if self.var_params[i] == 'gap':
                    gap_str = self._get_gap_str(parameter)
                    fname += '_gap{}'.format(gap_str)
                elif self.var_params[i] == 'phase':
                    phase_str = self._get_phase_str(parameter)
                    fname += '_phase{}'.format(phase_str)
                elif self.var_params[i] == 'width':
                    width = parameter
                    fname += '_width{}'.format(width)
            _save_pickle(fdata, fname, overwrite=True)

    def calc_fields(self):
        rx = _np.linspace(-self.rx_max, self.rx_max, self.rx_nrpts)
        rz = _np.linspace(-self.rz_max, self.rz_max, self.rz_nrpts)

        self._get_field_roll_off(rx=rx)
        self._get_field_on_axis(rz=rz)
        self._get_field_on_trajectory()

        self._save_data()

    # def run_plot_data(self, scan, values):
    #     data_plot = dict()
    #     fname = self.FOLDER_DATA
    #     gap_str = self._get_gap_str(gap)
    #     for width in widths:
    #         fname += 'field_data_gap{}_width{}'.format(gap_str, width)
    #         fdata = _load_pickle(fname)
    #         data_plot[width] = fdata

    @staticmethod
    def _get_field_component_idx(field_component):
        components = {'bx': 0, 'by': 1, 'bz': 2}
        return components[field_component]

    @staticmethod
    def _get_gap_str(gap):
        gap_str = '{:04.1f}'.format(gap).replace('.', 'p')
        return gap_str

    @staticmethod
    def _get_phase_str(phase):
        phase_str = '{:+07.3f}'.format(phase).replace('.', 'p')
        phase_str = phase_str.replace('+', 'pos').replace('-', 'neg')
        return phase_str
