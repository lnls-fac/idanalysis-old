#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
from idanalysis import IDKickMap as _IDKickMap

from mathphys.functions import save_pickle as _save_pickle
from mathphys.functions import load_pickle as _load_pickle

import utils


class AnalysisFromRadia:

    def __init__(self):
        # Model attributes
        self.rt_max = None  # [mm]
        self.rt_nrpts = None
        self.rz_max = None  # [mm]
        self.rz_nrpts = None
        self.roll_off_rt = utils.ROLL_OFF_RT  # [mm]
        self.model_params = dict()
        self.data = dict()
        self._models = dict()

        # Trajectory attributes
        self._idkickmap = None
        self.beam_energy = utils.BEAM_ENERGY  # [Gev]
        self.rk_s_step = utils.DEF_RK_S_STEP  # [mm]
        self.traj_init_rz = None  # [mm]
        self.traj_max_rz = None  # [mm]
        self.kmap_idlen = None  # [m]
        self.gridx = None
        self.gridy = None

        self.FOLDER_DATA = './results/model/data/'

    @property
    def idkickmap(self):
        """Return an object of IDKickMap class.

        Returns:
            IDKickMap object:
        """
        return self._idkickmap

    @property
    def models(self):
        """Return a dictionary with all ID models.

        Returns:
            Dictionary: A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        return self._models

    @models.setter
    def models(self, ids):
        """Set models attribute.

        Args:
            models (dictionary): A dictionary in which keys are
                the variable parameters and the values are the models.
                example: {(2, 40): model1, (2, 45): model2}, the keys could be
                gap and width for example.
        """
        self._models = ids

    @idkickmap.setter
    def idkickmap(self, id):
        """Set idkickmap config for trajectory."""
        self._idkickmap = _IDKickMap()
        self._idkickmap.radia_model = id
        self._idkickmap.beam_energy = self.beam_energy
        self._idkickmap.rk_s_step = self.rk_s_step
        self._idkickmap.traj_init_rz = self.traj_init_rz
        self._idkickmap.traj_rk_min_rz = self.traj_max_rz
        self._idkickmap.kmap_idlen = self.kmap_idlen
        self.traj_init_rz = self.traj_init_rz
        self.traj_max_rz = self.traj_max_rz
        self.kmap_idlen = self.kmap_idlen

    def generate_kickmap(self, key, id):
        width = key[0][1]
        phase = key[1][1]
        gap = key[2][1]
        self.idkickmap = id
        self.idkickmap.fmap_calc_kickmap(posx=self.gridx, posy=self.gridy)
        fname = self._get_kmap_filename(width=width, phase=phase, gap=gap)
        self.idkickmap.save_kickmap_file(kickmap_filename=fname)

    def _create_models(self):
        models_ = dict()
        for width in utils.widths:
            for phase in utils.phases:
                for gap in utils.gaps:
                    print(
                        f'creating model for gap {gap} mm, phase {phase} mm' +
                        f' and width {width} mm')
                    id = utils.generate_radia_model(
                        width=width,
                        phase=phase,
                        gap=gap,
                        solve=utils.SOLVE_FLAG)
                    key = (('width', width), ('phase', phase), ('gap', gap))
                    models_[(key)] = id
        self.models = models_

    def _get_field_roll_off(self, rt, peak_idx=0):
        """Calculate the roll-off of a field component.

        Args:
            rt (numpy 1D array): array with positions where the field will be
                calculated
            peak_idx (int): Peak index where the roll-off will be calculated.
                Defaults to 0.
        """
        field_component = utils.field_component
        b_ = dict()
        rt_dict = dict()
        roll_off = dict()
        comp_idx = self._get_field_component_idx(field_component)
        for var_params, id in self.models.items():
            period = id.period_length
            rz = _np.linspace(-period/2, period/2, 201)
            field = id.get_field(0, 0, rz)
            b = field[:, comp_idx]
            b_max_idx = _np.argmax(b)
            rz_at_max = rz[b_max_idx] + peak_idx*period
            field = id.get_field(rt, 0, rz_at_max)
            b = field[:, comp_idx]

            roff_idx = _np.argmin(_np.abs(rt-self.roll_off_rt))
            rt0_idx = _np.argmin(_np.abs(rt))
            roff = _np.abs(b[roff_idx]/b[rt0_idx]-1)

            b_[var_params] = b
            rt_dict[var_params] = rt
            roll_off[var_params] = roff

        self.data['rolloff_rt'] = rt_dict
        self.data['rolloff_{}'.format(field_component)] = b_
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
        for var_params, id in self.models.items():
            # create IDKickMap and calc trajectory
            self.idkickmap = id
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

    def _generate_field_data_fname(self, keys):
        fname = self.FOLDER_DATA
        sulfix = 'field_data'
        for parameter in keys:
            if parameter[0] == 'width':
                width = parameter[1]
                fname = fname.replace(
                    'data/', 'widths/width_{}/'.format(width))
                sulfix += '_width{}'.format(width)

            if parameter[0] == 'phase':
                phase_str = self._get_phase_str(parameter[1])
                fname += 'phases/phase_{}/'.format(phase_str)
                sulfix += '_phase{}'.format(phase_str)

            if parameter[0] == 'gap':
                gap_str = self._get_gap_str(parameter[1])
                fname += 'gap_{}/'.format(gap_str)
                sulfix += '_gap{}'.format(gap_str)
        return fname + sulfix

    def _get_field_data_fname(self, width, phase, gap):
        fname = self.FOLDER_DATA
        sulfix = 'field_data'

        fname = fname.replace(
            'data/', 'widths/width_{}/'.format(width))
        sulfix += '_width{}'.format(width)

        phase_str = self._get_phase_str(phase)
        fname += 'phases/phase_{}/'.format(phase_str)
        sulfix += '_phase{}'.format(phase_str)

        gap_str = self._get_gap_str(gap)
        fname += 'gap_{}/'.format(gap_str)
        sulfix += '_gap{}'.format(gap_str)
        return fname + sulfix

    def _save_field_data(self):
        #  re-organize data
        for keys in list(self.models.keys()):
            fdata = dict()
            for info, value in self.data.items():
                fdata[info] = value[keys]
            # get filename
            fname = self._generate_field_data_fname(keys=keys)
            print(fname)
            _save_pickle(fdata, fname, overwrite=True, makedirs=True)

    def _get_kmap_filename(self, width, gap, phase):
        fpath = self.FOLDER_DATA + 'kickmaps/'
        fpath = fpath.replace('model/data/', 'model/')
        gap_str = self._get_gap_str(gap)
        phase_str = self._get_phase_str(phase)
        width = width
        fname = fpath + f'kickmap-ID_width{width}_phase{phase_str}' +\
            f'_gap{gap_str}.txt'
        return fname

    def run_calc_fields(self):

        self._create_models()

        rt = _np.linspace(-self.rt_max, self.rt_max, self.rt_nrpts)
        rz = _np.linspace(-self.rz_max, self.rz_max, self.rz_nrpts)

        self._get_field_roll_off(rt=rt)
        self._get_field_on_axis(rz=rz)
        self._get_field_on_trajectory()

        self._save_field_data()

    def run_plot_data(self, width=0, phase=0, gap=0):
        data_plot = dict()
        if utils.var_param == 'gap':
            for gap_ in utils.gaps:
                fname = self._get_field_data_fname(width, phase, gap_)
                fdata = _load_pickle(fname)
                data_plot[gap_] = fdata
        if utils.var_param == 'phase':
            for phase_ in utils.phases:
                fname = self._get_field_data_fname(width, phase_, gap)
                fdata = _load_pickle(fname)
                data_plot[phase_] = fdata
        if utils.var_param == 'width':
            for width_ in utils.widths:
                fname = self._get_field_data_fname(width_, phase, gap)
                fdata = _load_pickle(fname)
                data_plot[phase_] = fdata

        self._plot_field_on_axis(data=data_plot)
        self._plot_rk_traj(data=data_plot)
        self._plot_field_roll_off(data=data_plot)

    def run_generate_kickmap(self):
        if self.models:
            print("models ready")
        else:
            self._create_models()
            print("models ready")

        for key, id in self.models.items():
            print(f'calc kickmap for gap {key[2][1]} mm, ' +
                  f'phase {key[1][1]} mm ' +
                  f'and width {key[0][1]} mm')
            self.generate_kickmap(key, id)

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

    @staticmethod
    def _plot_field_on_axis(data):
        _plt.figure(1)
        var_parameters = list(data.keys())
        for parameter in var_parameters:
            label = utils.var_param + ' {}'.format(parameter)
            by = data[parameter]['onaxis_by']
            rz = data[parameter]['onaxis_rz']
            _plt.plot(rz, by, label=label)
        _plt.xlabel('z [mm]')
        _plt.ylabel('By [T]')
        _plt.legend()
        _plt.grid()
        _plt.savefig(utils.FOLDER_DATA + 'general/field-profile', dpi=300)
        _plt.show()

    @staticmethod
    def _plot_rk_traj(data):
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        var_parameters = list(data.keys())
        for i, parameter in enumerate(var_parameters):
            s = data[parameter]['ontraj_s']
            rx = data[parameter]['ontraj_rx']
            ry = data[parameter]['ontraj_ry']
            px = 1e6*data[parameter]['ontraj_px']
            py = 1e6*data[parameter]['ontraj_py']
            label = utils.var_param + ' {}'.format(parameter)

            _plt.figure(1)
            _plt.plot(s, 1e3*rx, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('x [um]')

            _plt.figure(2)
            _plt.plot(s, 1e3*ry, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('y [um]')
            _plt.legend()

            _plt.figure(3)
            _plt.plot(s, px, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('px [urad]')

            _plt.figure(4)
            _plt.plot(s, py, color=colors[i], label=label)
            _plt.xlabel('s [mm]')
            _plt.ylabel('py [urad]')
        sulfix = ['traj-rx', 'traj-ry', 'traj-px', 'traj-py']
        for i in [1, 2, 3, 4]:
            _plt.figure(i)
            _plt.legend()
            _plt.grid()
            _plt.savefig(utils.FOLDER_DATA + 'general/' + sulfix[i-1], dpi=300)
        _plt.show()

    @staticmethod
    def _plot_field_roll_off(data):
        _plt.figure(1)
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        var_parameters = list(data.keys())
        for i, parameter in enumerate(var_parameters):
            by_ = data[parameter]['rolloff_by']
            rt = data[parameter]['rolloff_rt']
            by = _np.array(by_)
            rtp_idx = _np.argmin(_np.abs(rt - utils.ROLL_OFF_RT))
            rt0_idx = _np.argmin(_np.abs(rt))
            roff = _np.abs(by[rtp_idx]/by[rt0_idx]-1)
            label = utils.var_param +\
                " {}, roll-off = {:.2f} %".format(parameter, 100*roff)
            irt0 = _np.argmin(_np.abs(rt))
            by0 = by[irt0]
            roll_off = 100*(by/by0 - 1)
            _plt.plot(rt, roll_off, '.-', label=label, color=colors[i])
        _plt.xlabel('x [mm]')
        _plt.ylabel('roll off [%]')
        _plt.xlim(-6, 6)
        _plt.ylim(-1.2, 0.02)
        _plt.title('Field roll-off at x = {} mm'.format(utils.ROLL_OFF_RT))
        _plt.legend()
        _plt.grid()
        _plt.savefig(utils.FOLDER_DATA + 'general/field-rolloff', dpi=300)
        _plt.show()
