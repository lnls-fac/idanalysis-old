#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
from idanalysis import IDKickMap as _IDKickMap
from scipy import integrate as _integrate


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
    def _config_traj(self):
        """Set idkickmap config for trajectory."""
        self.idkickmap = _IDKickMap()
        self.idkickmap.beam_energy = self.beam_energy
        self.idkickmap.rk_s_step = self.rk_s_step
        self.idkickmap.traj_init_rz = self.traj_init_rz
        self.idkickmap.traj_rk_min_rz = self.traj_max_rz

    def get_field_roll_off(self, rx, peak_idx=0, field_component='by'):
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
            rz = _np.linspace(-period/2, period/2, 100)
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

    def get_field_on_axis(self, rz):
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

    def calc_fields(self):
        rx = _np.linspace(-self.rx_max, self.rx_max, self.rx_nrpts)
        rz = _np.linspace(-self.rz_max, self.rz_max, self.rz_nrpts)

        self.get_field_roll_off(rx=rx)

    @staticmethod
    def _get_field_component_idx(field_component):
        components = {'bx': 0, 'by': 1, 'bz': 2}
        return components[field_component]
