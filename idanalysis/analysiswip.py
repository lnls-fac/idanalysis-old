#!/usr/bin/env python-sirius

import numpy as _np
import matplotlib.pyplot as _plt
from scipy.optimize import curve_fit as _curve_fit

import pyaccel
import pymodels

from fieldmaptrack import Beam

from mathphys.functions import save_pickle as _save_pickle
from mathphys.functions import load_pickle as _load_pickle

from idanalysis import IDKickMap as _IDKickMap
from idanalysis import optics as optics
from idanalysis import orbcorr as orbcorr

from apsuite.dynap import DynapXY

import utils


class Tools:
    class CALC_TYPES:
        nominal = 0
        nonsymmetrized = 1
        symmetrized = 2

    @staticmethod
    def _get_data_path(width, phase, gap):
        fpath = utils.FOLDER_DATA
        fpath = fpath.replace(
            'data/', 'data/widths/width_{}/'.format(width))

        phase_str = Tools._get_phase_str(phase)
        fpath += 'phases/phase_{}/'.format(phase_str)

        gap_str = Tools._get_gap_str(gap)
        fpath += 'gap_{}/'.format(gap_str)
        return fpath

    @staticmethod
    def _get_kmap_filename(
            width, gap, phase,
            shift_flag=False, filter_flag=False, linear=False):
        fpath = utils.FOLDER_DATA + 'kickmaps/'
        fpath = fpath.replace('model/data/', 'model/')
        gap_str = Tools._get_gap_str(gap)
        phase_str = Tools._get_phase_str(phase)
        width = width
        fname = fpath + f'kickmap-ID_width{width}_phase{phase_str}' +\
            f'_gap{gap_str}.txt'
        if linear:
            fname = fname.replace('.txt', '-linear.txt')
            return fname
        if shift_flag:
            fname = fname.replace('.txt', '-shifted_on_axis.txt')
        if filter_flag:
            fname = fname.replace('.txt', '-filtered.txt')
        return fname

    @staticmethod
    def _create_model_ids(
            fname,
            rescale_kicks=utils.RESCALE_KICKS,
            rescale_length=utils.RESCALE_LENGTH):
        ids = utils.create_ids(
            fname, rescale_kicks=rescale_kicks,
            rescale_length=rescale_length)
        model = pymodels.si.create_accelerator(ids=ids)
        twiss, *_ = pyaccel.optics.calc_twiss(model, indices='closed')
        print('length : {:.4f} m'.format(model.length))
        print('tunex  : {:.6f}'.format(twiss.mux[-1]/2/_np.pi))
        print('tuney  : {:.6f}'.format(twiss.muy[-1]/2/_np.pi))
        return model, ids

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
    def _mkdir_function(mypath):
        from errno import EEXIST
        from os import makedirs, path

        try:
            makedirs(mypath)
        except OSError as exc:
            if exc.errno == EEXIST and path.isdir(mypath):
                pass
            else:
                raise


class FieldAnalysisFromRadia(Tools):

    def __init__(self):
        # Model attributes
        self.rt_field_max = None  # [mm]
        self.rt_field_nrpts = None
        self.rz_field_max = None  # [mm]
        self.rz_field_nrpts = None
        self.roll_off_rt = utils.ROLL_OFF_RT  # [mm]
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

    def _generate_kickmap(self, key, id):
        width = key[0][1]
        phase = key[1][1]
        gap = key[2][1]
        self.idkickmap = id
        self.idkickmap.fmap_calc_kickmap(posx=self.gridx, posy=self.gridy)
        fname = self._get_kmap_filename(
            width=width, phase=phase, gap=gap)
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
            if field_component == 'bx':
                field = id.get_field(0, rt, rz_at_max)
            elif field_component == 'by':
                field = id.get_field(rt, 0, rz_at_max)
            else:
                raise ValueError
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
                    'data/', 'data/widths/width_{}/'.format(width))
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
        fpath = self._get_data_path(width=width, phase=phase, gap=gap)

        sulfix = 'field_data'
        sulfix += '_width{}'.format(width)

        phase_str = self._get_phase_str(phase)
        sulfix += '_phase{}'.format(phase_str)

        gap_str = self._get_gap_str(gap)
        sulfix += '_gap{}'.format(gap_str)
        return fpath + sulfix

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

    def run_calc_fields(self):

        self._create_models()

        rt = _np.linspace(
            -self.rt_field_max, self.rt_field_max, self.rt_field_nrpts)
        rz = _np.linspace(
            -self.rz_field_max, self.rz_field_max, self.rz_field_nrpts)

        self._get_field_roll_off(rt=rt)
        self._get_field_on_axis(rz=rz)
        self._get_field_on_trajectory()

        self._save_field_data()

    def _get_data_plot(self, width=0, phase=0, gap=0):
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
                data_plot[width_] = fdata
        return data_plot

    def run_plot_data(self, width=0, phase=0, gap=0):
        data_plot = self._get_data_plot(width=width, phase=phase, gap=gap)
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
            self._generate_kickmap(key, id)

    def _plot_field_on_axis(self, data):
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        field_component = utils.field_component
        _plt.figure(1)
        output_dir = utils.FOLDER_DATA + 'general'
        self._mkdir_function(output_dir)
        var_parameters = list(data.keys())
        for i, parameter in enumerate(var_parameters):
            label = utils.var_param + ' {}'.format(parameter)
            b = data[parameter]['onaxis_{}'.format(field_component)]
            rz = data[parameter]['onaxis_rz']
            _plt.plot(rz, b, label=label, color=colors[i])
        _plt.xlabel('z [mm]')
        _plt.ylabel('{} [T]'.format(field_component))
        _plt.legend()
        _plt.grid()
        _plt.savefig(output_dir + '/field-profile', dpi=300)
        _plt.show()

    def _plot_rk_traj(self, data):
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        var_parameters = list(data.keys())
        output_dir = utils.FOLDER_DATA + 'general'
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
        sulfix = ['/traj-rx', '/traj-ry', '/traj-px', '/traj-py']
        for i in [1, 2, 3, 4]:
            _plt.figure(i)
            _plt.legend()
            _plt.grid()
            _plt.savefig(output_dir + sulfix[i-1], dpi=300)
        _plt.show()

    def _read_data_roll_off(self, data, parameter):
        field_component = utils.field_component
        b = data[parameter]['rolloff_{}'.format(field_component)]
        rt = data[parameter]['rolloff_rt']
        rtp_idx = _np.argmin(_np.abs(rt - utils.ROLL_OFF_RT))
        rt0_idx = _np.argmin(_np.abs(rt))
        roff = _np.abs(b[rtp_idx]/b[rt0_idx]-1)
        b0 = b[rt0_idx]
        roll_off = 100*(b/b0 - 1)
        return rt, b, roll_off, roff

    def _plot_field_roll_off(self, data):
        field_component = utils.field_component
        _plt.figure(1)
        output_dir = utils.FOLDER_DATA + 'general'
        colors = ['b', 'g', 'y', 'C1', 'r', 'k']
        var_parameters = list(data.keys())
        for i, parameter in enumerate(var_parameters):
            rt, b, roll_off, roff = self._read_data_roll_off(data, parameter)
            label = utils.var_param +\
                " {}, roll-off = {:.2f} %".format(parameter, 100*roff)
            _plt.plot(rt, roll_off, '.-', label=label, color=colors[i])
        if field_component == 'by':
            _plt.xlabel('x [mm]')
        else:
            _plt.xlabel('y [mm]')
        _plt.ylabel('roll off [%]')
        _plt.xlim(-utils.ROLL_OFF_RT, utils.ROLL_OFF_RT)
        _plt.ylim(-3, 0.02)
        if field_component == 'by':
            _plt.title('Field roll-off at x = {} mm'.format(utils.ROLL_OFF_RT))
        elif field_component == 'bx':
            _plt.title('Field roll-off at y = {} mm'.format(utils.ROLL_OFF_RT))
        _plt.legend()
        _plt.grid()
        _plt.savefig(output_dir + '/field-rolloff', dpi=300)
        _plt.show()

    def _get_planar_id_features(self,
                                width=0, phase=0, gap=0,
                                plot_flag=False):

        def cos(x, b0, kx):
            b = b0*_np.cos(kx*x)
            return b

        data = self._get_data_plot(width=width, phase=phase, gap=gap)
        rt, b, *_ = self._read_data_roll_off(data, width)
        idxi = _np.argmin(_np.abs(rt+1))
        idxf = _np.argmin(_np.abs(rt-1))
        rt = rt[idxi:idxf]
        b = b[idxi:idxf]
        opt = _curve_fit(cos, rt, b)[0]
        k = opt[1]
        b0 = opt[0]
        print('field amplitude = {:.3f} T'.format(b0))
        if plot_flag:
            _plt.plot(rt, b, label='model')
            b_fitted = cos(rt, opt[0], opt[1])
            _plt.plot(rt, b_fitted, label='fitting')
            _plt.xlabel('y [mm]')
            _plt.ylabel('B [T]')
            _plt.title('Field Roll-off fitting')
            _plt.grid()
            _plt.legend()
            _plt.show()

        kz = 2*_np.pi/(utils.ID_PERIOD*1e-3)
        kx = k*1e3
        ky = _np.sqrt(kx**2+kz**2)
        print('kx = {:.2f}'.format(kx))
        print('ky = {:.2f}'.format(ky))
        print('kz = {:.2f}'.format(kz))

        return b0, kx, ky, kz

    def _get_id_quad_coef(self,
                          width=0, phase=0, gap=0,
                          plot_flag=False):
        id_len = utils.SIMODEL_ID_LEN
        b0, kx, ky, kz = self._get_planar_id_features(width, phase, gap,
                                                      plot_flag=plot_flag)
        beam = Beam(energy=3)
        brho = beam.brho
        phase = phase*1e-3
        factor = 1 + _np.cos(kz*phase)
        factor -= (kx**2)/(kz**2+kx**2)*(1-_np.cos(kz*phase))
        a = (1/(kz**2))*id_len*(b0**2)/(4*brho**2)
        if utils.field_component == 'by':
            coefx = a*factor*kx**2
            coefy = -a*2*ky**2
        elif utils.field_component == 'bx':
            coefx = -a*factor*ky**2
            coefy = a*2*kx**2

        return coefx, coefy

    def get_id_estimated_focusing(self, betax, betay,
                                  width=0, phase=0, gap=0, plot_flag=False):

        id_len = utils.SIMODEL_ID_LEN
        coefx, coefy = self._get_id_quad_coef(width=width, phase=phase,
                                              gap=gap, plot_flag=plot_flag)
        dtunex = -id_len*coefx*betax/(4*_np.pi)
        dtuney = -id_len*coefy*betay/(4*_np.pi)

        print('horizontal quadrupolar term: {:.5f}'.format(coefx))
        print('vertical quadrupolar term: {:.5f}'.format(coefy))
        print('horizontal delta tune: {:.5f}'.format(dtunex))
        print('vertical delta tune: {:.5f}'.format(dtuney))

        return coefx, coefy, dtunex, dtuney

    def generate_linear_kickmap(self, width=0, phase=0, gap=0,
                                cxy=0, cyx=0):
        beam = Beam(energy=3)
        brho = beam.brho
        idkickmap = _IDKickMap()
        cxx, cyy = self._get_id_quad_coef(width=width,
                                          phase=phase, gap=gap)

        idkickmap.generate_linear_kickmap(brho=brho, posx=self.gridx,
                                          posy=self.gridy, cxx=cxx, cyy=cyy,
                                          cxy=cxy, cyx=cyx)
        fname = self._get_kmap_filename(
            width=width, phase=phase, gap=gap)
        fname = fname.replace('.txt', '-linear.txt')
        idkickmap.kmap_idlen = utils.ID_KMAP_LEN
        idkickmap.save_kickmap_file(kickmap_filename=fname)


class AnalysisKickmap(Tools):

    def __init__(self):
        self._idkickmap = None
        self.save_flag = False
        self.plot_flag = False
        self.filter_flag = False
        self.shift_flag = False
        self.FOLDER_DATA = utils.FOLDER_DATA

    def _get_figname_plane(self, kick_plane, var='X',
                           width=None, phase=None, gap=None, linear=False):
        fpath = utils.FOLDER_DATA
        fpath = fpath.replace('data/', 'data/general/')

        if utils.var_param == 'gap':
            phase_str = self._get_phase_str(phase)
            fname = fpath + 'kick{}-vs-{}-width{}mm-phase{}.png'.format(
                kick_plane, var.lower(), width, phase_str)

        if utils.var_param == 'phase':
            gap_str = self._get_gap_str(gap)
            fname = fpath + 'kick{}-vs-{}-width{}mm-gap{}.png'.format(
                kick_plane, var.lower(), width, gap_str)

        if utils.var_param == 'width':
            phase_str = self._get_phase_str(phase)
            gap_str = self._get_gap_str(gap)
            fname = fpath + 'kick{}-vs-{}-phase{}-gap{}.png'.format(
                kick_plane, var.lower(), phase_str, gap_str)

        return fname

    def _get_figname_allplanes(self, kick_plane, var='X',
                               width=None, phase=None, gap=None, linear=False):
        fname = utils.FOLDER_DATA

        sulfix = 'kick{}-vs-{}-all-planes'.format(
            kick_plane.lower(), var.lower())

        fname = fname.replace(
            'data/', 'data/widths/width_{}/'.format(width))

        phase_str = self._get_phase_str(phase)
        fname += 'phases/phase_{}/'.format(phase_str)

        gap_str = self._get_gap_str(gap)
        fname += 'gap_{}/'.format(gap_str)

        fname += sulfix
        return fname

    def run_shift_kickmap(self, widths, phases, gaps):
        for width in widths:
            for phase in phases:
                for gap in gaps:
                    fname = self._get_kmap_filename(
                        width=width, phase=phase, gap=gap)
                    self._idkickmap = _IDKickMap(
                        kmap_fname=fname, shift_on_axis=True)
                    fname = fname.replace('.txt', '-shifted_on_axis.txt')
                    self._idkickmap.save_kickmap_file(fname)

    def run_filter_kickmap(self, widths, phases, gaps,
                           rx, filter_order=4, is_shifted=True):
        for width in widths:
            for phase in phases:
                for gap in gaps:
                    fname = self._get_kmap_filename(
                        width=width, phase=phase, gap=gap,
                        shift_flag=is_shifted)
                    self._idkickmap = _IDKickMap(fname)
                    self._idkickmap.filter_kmap(
                        posx=rx, order=filter_order, plot_flag=False)
                    self._idkickmap.kmap_idlen = utils.ID_KMAP_LEN
                    fname = fname.replace('.txt', '-filtered.txt')
                    self._idkickmap.save_kickmap_file(fname)

    def _calc_idkmap_kicks(self, plane_idx=0, var='X'):
        beam = Beam(energy=3)
        brho = beam.brho
        rx0 = self._idkickmap.posx
        ry0 = self._idkickmap.posy
        fposx = self._idkickmap.fposx
        fposy = self._idkickmap.fposy
        kickx = self._idkickmap.kickx
        kicky = self._idkickmap.kicky
        if var.lower() == 'x':
            rxf = fposx[plane_idx, :]
            ryf = fposy[plane_idx, :]
            pxf = kickx[plane_idx, :]/brho**2
            pyf = kicky[plane_idx, :]/brho**2
        elif var.lower() == 'y':
            rxf = fposx[:, plane_idx]
            ryf = fposy[:, plane_idx]
            pxf = kickx[:, plane_idx]/brho**2
            pyf = kicky[:, plane_idx]/brho**2

        return rx0, ry0, pxf, pyf, rxf, ryf

    def check_kick_at_plane(
            self, width=None, phase=None, gap=None,
            planes=['X', 'Y'], kick_planes=['X', 'Y'], linear=False):

        if utils.var_param == 'width':
            var_params = utils.widths
        if utils.var_param == 'phase':
            var_params = utils.phases
        if utils.var_param == 'gap':
            var_params = utils.gaps

        for var in planes:
            for kick_plane in kick_planes:
                colors = ['b', 'g', 'y', 'C1', 'r', 'k']
                for var_param, color in zip(var_params, colors):

                    if utils.var_param == 'width':
                        fname = self._get_kmap_filename(
                            width=var_param, gap=gap, phase=phase,
                            shift_flag=self.shift_flag,
                            filter_flag=self.filter_flag,
                            linear=linear)
                        fname_fig = self._get_figname_plane(
                            kick_plane=kick_plane, var=var,
                            width=var_param, gap=gap, phase=phase,
                            linear=linear)
                    if utils.var_param == 'phase':
                        fname = self._get_kmap_filename(
                            width=width, gap=gap, phase=var_param,
                            shift_flag=self.shift_flag,
                            filter_flag=self.filter_flag,
                            linear=linear)
                        fname_fig = self._get_figname_plane(
                            kick_plane=kick_plane, var=var,
                            width=width, gap=gap, phase=var_param,
                            linear=linear)
                    if utils.var_param == 'gap':
                        fname = self._get_kmap_filename(
                            width=width, gap=var_param, phase=phase,
                            shift_flag=self.shift_flag,
                            filter_flag=self.filter_flag,
                            linear=linear)
                        fname_fig = self._get_figname_plane(
                            kick_plane=kick_plane, var=var,
                            width=width, gap=var_param, phase=phase,
                            linear=linear)

                    self._idkickmap = _IDKickMap(fname)
                    if var.lower() == 'x':
                        pos_zero_idx = list(self._idkickmap.posy).index(0)
                    elif var.lower() == 'y':
                        pos_zero_idx = list(self._idkickmap.posx).index(0)
                    rx0, ry0, pxf, pyf, *_ = self._calc_idkmap_kicks(
                                                    plane_idx=pos_zero_idx,
                                                    var=var)
                    pxf *= utils.RESCALE_KICKS
                    pyf *= utils.RESCALE_KICKS

                    pf, klabel =\
                        (pxf, 'px') if kick_plane.lower() == 'x' else (
                         pyf, 'py')

                    if var.lower() == 'x':
                        r0, xlabel, rvar = (rx0, 'x0 [mm]', 'y')
                        pfit = _np.polyfit(r0, pf, 21)
                    else:
                        r0, xlabel, rvar = (ry0, 'y0 [mm]', 'x')
                        pfit = _np.polyfit(r0, pf, 11)
                    pf_fit = _np.polyval(pfit, r0)

                    label = utils.var_param + ' = {} mm'.format(var_param)
                    _plt.figure(1)
                    _plt.plot(
                        1e3*r0, 1e6*pf, '.-', color=color, label=label)
                    _plt.plot(
                        1e3*r0, 1e6*pf_fit, '-', color=color, alpha=0.6)
                    print('kick plane: ', kick_plane)
                    print('plane: ', var)
                    print('Fitting:')
                    print(pfit[::-1])

                _plt.figure(1)
                _plt.xlabel(xlabel)
                _plt.ylabel('final {} [urad]'.format(klabel))
                _plt.title('Kick{} for gap {} mm, at pos{} {:+.3f} mm'.format(
                    kick_plane.upper(), gap, rvar, 0))
                _plt.legend()
                _plt.grid()
                _plt.tight_layout()
                if self.save_flag:
                    _plt.savefig(fname_fig, dpi=300)
                if self.plot_flag:
                    _plt.show()
                _plt.close()

    def check_kick_all_planes(
            self, width=None, phase=None, gap=None,
            planes=['X', 'Y'], kick_planes=['X', 'Y'], linear=False):
        for var in planes:
            for kick_plane in kick_planes:
                fname = self._get_kmap_filename(width=width,
                                                phase=phase,
                                                gap=gap,
                                                linear=linear)
                self._idkickmap = _IDKickMap(fname)
                fname_fig = self._get_figname_allplanes(
                    width=width, phase=phase, gap=gap,
                    var=var, kick_plane=kick_plane,
                    linear=linear)

                if var.lower() == 'x':
                    kmappos = self._idkickmap.posy
                else:
                    kmappos = self._idkickmap.posx

                for plane_idx, pos in enumerate(kmappos):
                    if pos < 0:
                        continue
                    rx0, ry0, pxf, pyf, *_ = self._calc_idkmap_kicks(
                        var=var, plane_idx=plane_idx)
                    pxf *= utils.RESCALE_KICKS
                    pyf *= utils.RESCALE_KICKS
                    pf, klabel =\
                        (pxf, 'px') if kick_plane.lower() == 'x' else (
                         pyf, 'py')
                    if var.lower() == 'x':
                        r0, xlabel, rvar = (rx0, 'x0 [mm]', 'y')
                    else:
                        r0, xlabel, rvar = (ry0, 'y0 [mm]', 'x')
                    rvar = 'y' if var.lower() == 'x' else 'x'
                    label = 'pos{} = {:+.3f} mm'.format(rvar, 1e3*pos)
                    _plt.plot(
                        1e3*r0, 1e6*pf, '.-', label=label)
                    _plt.xlabel(xlabel)
                    _plt.ylabel('final {} [urad]'.format(klabel))
                    _plt.title(
                        'Kicks for gap {} mm, width {} mm'.format(gap, width))
                _plt.legend()
                _plt.grid()
                _plt.tight_layout()
                if self.save_flag:
                    _plt.savefig(fname_fig, dpi=300)
                if self.plot_flag:
                    _plt.show()
                _plt.close()

    def check_kick_at_plane_trk(self, width, gap, phase, linear=False):
        fname = self._get_kmap_filename(width=width, gap=gap,
                                        phase=phase, linear=linear)
        self._idkickmap = _IDKickMap(fname)
        plane_idx = list(self._idkickmap.posy).index(0)
        out = self._calc_idkmap_kicks(plane_idx=plane_idx)
        rx0, ry0 = out[0], out[1]
        pxf, pyf = out[2], out[3]
        rxf, ryf = out[4], out[5]
        pxf *= utils.RESCALE_KICKS
        pyf *= utils.RESCALE_KICKS

        # lattice with IDs
        model, _ = self._create_model_ids(fname)

        famdata = pymodels.si.get_family_data(model)

        # shift model
        idx = famdata[utils.ID_FAMNAME]['index']
        idx_begin = idx[0][0]
        idx_end = idx[0][-1]
        idx_dif = idx_end - idx_begin

        model = pyaccel.lattice.shift(model, start=idx_begin)

        rxf_trk, ryf_trk = _np.ones(len(rx0)), _np.ones(len(rx0))
        pxf_trk, pyf_trk = _np.ones(len(rx0)), _np.ones(len(rx0))
        for i, x0 in enumerate(rx0):
            coord_ini = _np.array([x0, 0, 0, 0, 0, 0])
            coord_fin, *_ = pyaccel.tracking.line_pass(
                model, coord_ini, indices='open')

            rxf_trk[i] = coord_fin[0, idx_dif+1]
            ryf_trk[i] = coord_fin[2, idx_dif+1]
            pxf_trk[i] = coord_fin[1, idx_dif+1]
            pyf_trk[i] = coord_fin[3, idx_dif+1]

        _plt.plot(
            1e3*rx0, 1e6*(rxf - rx0), '.-', color='C1', label='Pos X  kickmap')
        _plt.plot(1e3*rx0, 1e6*ryf, '.-', color='b', label='Pos Y  kickmap')
        _plt.plot(
            1e3*rx0, 1e6*(rxf_trk - rx0), 'o', color='C1',
            label='Pos X  tracking')
        _plt.plot(
            1e3*rx0, 1e6*ryf_trk, 'o', color='b', label='Pos Y  tracking')
        _plt.xlabel('x0 [mm]')
        _plt.ylabel('final dpos [um]')
        _plt.title('dPos')
        _plt.legend()
        _plt.grid()
        _plt.show()

        _plt.plot(1e3*rx0, 1e6*pxf, '.-', color='C1', label='Kick X  kickmap')
        _plt.plot(1e3*rx0, 1e6*pyf, '.-', color='b', label='Kick Y  kickmap')
        _plt.plot(
            1e3*rx0, 1e6*pxf_trk, 'o', color='C1', label='Kick X  tracking')
        _plt.plot(
            1e3*rx0, 1e6*pyf_trk, 'o', color='b', label='Kick Y  tracking')
        _plt.xlabel('x0 [mm]')
        _plt.ylabel('final px [urad]')
        _plt.title('Kicks')
        _plt.legend()
        _plt.grid()
        _plt.show()


class AnalysisEffects(Tools):

    def __init__(self):
        self._idkickmap = None
        self._model_id = None
        self._ids = None
        self._twiss0 = None
        self._twiss1 = None
        self._twiss2 = None
        self._twiss3 = None
        self._beta_flag = None
        self._stg = None
        self.id_famname = utils.ID_FAMNAME
        self.fitted_model = False
        self.shift_flag = True
        self.filter_flag = False
        self.calc_type = 0
        self.corr_system = 'SOFB'
        self.orbcorr_plot_flag = False
        self.bb_plot_flag = False

    def _create_model_nominal(self):
        model0 = pymodels.si.create_accelerator()
        if self.fitted_model:
            model0 = \
                pymodels.si.fitted_models.vertical_dispersion_and_coupling(
                    model0)
        model0.cavity_on = False
        model0.radiation_on = 0
        return model0

    def _get_knobs_locs(self):

        straight_nr = dict()
        knobs = dict()
        locs_beta = dict()
        for id_ in self._ids:
            straight_nr_ = int(id_.subsec[2:4])

            # get knobs and beta locations
            if straight_nr_ is not None:
                _, knobs_, _ = optics.symm_get_knobs(
                        self._model_id, straight_nr_)
                locs_beta_ = optics.symm_get_locs_beta(knobs_)
            else:
                knobs_, locs_beta_ = None, None

            straight_nr[id_.subsec] = straight_nr_
            knobs[id_.subsec] = knobs_
            locs_beta[id_.subsec] = locs_beta_

        return knobs, locs_beta, straight_nr

    def _calc_coupling(self, x0, nturns=1000):
        coord_ini = _np.array([x0, 0, 0, 0, 0, 0])
        coord_fin, *_ = pyaccel.tracking.ring_pass(
            self._model_id, coord_ini, nr_turns=nturns,
            turn_by_turn=True, parallel=True)
        rx = coord_fin[0, :]
        ry = coord_fin[2, :]
        twiss, *_ = pyaccel.optics.calc_twiss(self._model_id)
        betax, betay = twiss.betax, twiss.betay  # Beta functions
        jx = 2/(betax[0]*nturns)*(_np.sum(rx)**2)
        jy = 2/(betay[0]*nturns)*(_np.sum(ry)**2)

        print('coupling k = {:.3f}'.format(jy/jx))
        return jy/jx

    def _calc_dtune_betabeat(self, twiss1):
        dtunex = (twiss1.mux[-1] - self._twiss0.mux[-1]) / 2 / _np.pi
        dtuney = (twiss1.muy[-1] - self._twiss0.muy[-1]) / 2 / _np.pi
        bbeatx = (twiss1.betax - self._twiss0.betax)/self._twiss0.betax
        bbeaty = (twiss1.betay - self._twiss0.betay)/self._twiss0.betay
        bbeatx *= 100
        bbeaty *= 100
        bbeatx_rms = _np.std(bbeatx)
        bbeaty_rms = _np.std(bbeaty)
        bbeatx_absmax = _np.max(_np.abs(bbeatx))
        bbeaty_absmax = _np.max(_np.abs(bbeaty))
        return (
            dtunex, dtuney, bbeatx, bbeaty,
            bbeatx_rms, bbeaty_rms, bbeatx_absmax, bbeaty_absmax)

    def _analysis_uncorrected_perturbation(self, plot_flag=True):
        self._twiss1, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices='closed')

        results = self._calc_dtune_betabeat(twiss1=self._twiss1)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]

        if plot_flag:

            print(f'dtunex: {dtunex:+.6f}')
            print(f'dtunex: {dtuney:+.6f}')
            txt = f'bbetax: {bbeatx_rms:04.1f} % rms, '
            txt += f'{bbeatx_absmax:04.1f} % absmax'
            print(txt)
            txt = f'bbetay: {bbeaty_rms:04.1f} % rms, '
            txt += f'{bbeaty_absmax:04.1f} % absmax'
            print(txt)

            labelx = f'X ({bbeatx_rms:.1f} % rms)'
            labely = f'Y ({bbeaty_rms:.1f} % rms)'
            _plt.plot(self._twiss1.spos, bbeatx,
                      color='b', alpha=1, label=labelx)
            _plt.plot(self._twiss1.spos, bbeaty,
                      color='r', alpha=0.8, label=labely)
            _plt.xlabel('spos [m]')
            _plt.ylabel('Beta Beat [%]')
            _plt.title('Beta Beating from ' + self.id_famname)
            _plt.legend()
            _plt.grid()
            _plt.show()

    def _plot_beta_beating(self, width, phase, gap):
        fpath = self._get_data_path(width=width, phase=phase, gap=gap)
        # Compare optics between nominal value and uncorrect optics due ID
        results = self._calc_dtune_betabeat(twiss1=self._twiss1)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
        print('Not symmetrized optics :')
        print(f'dtunex: {dtunex:+.2e}')
        print(f'dtuney: {dtuney:+.2e}')
        print(
            f'bbetax: {bbeatx_rms:04.3f} % rms,' +
            f' {bbeatx_absmax:04.3f} % absmax')
        print(
            f'bbetay: {bbeaty_rms:04.3f} % rms,' +
            f' {bbeaty_absmax:04.3f} % absmax')
        print()

        _plt.clf()

        label1 = {False: '-nominal', True: '-fittedmodel'}[self.fitted_model]

        _plt.figure(1)
        stg_tune = f'dtunex: {dtunex:+0.04f}\n' + f'dtuney: {dtuney:+0.04f}'
        labelx = f'X ({bbeatx_rms:.3f} % rms)'
        labely = f'Y ({bbeaty_rms:.3f} % rms)'
        _plt.plot(self._twiss0.spos, bbeatx,
                  color='b', alpha=1.0, label=labelx)
        _plt.plot(self._twiss0.spos, bbeaty,
                  color='r', alpha=0.8, label=labely)
        _plt.xlabel('spos [m]')
        _plt.ylabel('Beta Beating [%]')
        _plt.title('Tune shift' + '\n' + stg_tune)
        _plt.suptitle(self.id_famname + ' - Non-symmetrized optics')
        _plt.tight_layout()
        _plt.legend()
        _plt.grid()
        _plt.savefig(fpath + 'opt{}-ids-nonsymm'.format(label1), dpi=300)

        # Compare optics between nominal value and symmetrized optics
        results = self._calc_dtune_betabeat(twiss1=self._twiss2)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
        print('symmetrized optics but uncorrect tunes:')
        print(f'dtunex: {dtunex:+.0e}')
        print(f'dtuney: {dtuney:+.0e}')
        print(
            f'bbetax: {bbeatx_rms:04.3f} % rms,' +
            f' {bbeatx_absmax:04.3f} % absmax')
        print(
            f'bbetay: {bbeaty_rms:04.3f} % rms,' +
            f' {bbeaty_absmax:04.3f} % absmax')
        print()

        _plt.figure(2)
        labelx = f'X ({bbeatx_rms:.3f} % rms)'
        labely = f'Y ({bbeaty_rms:.3f} % rms)'
        _plt.plot(self._twiss0.spos, bbeatx,
                  color='b', alpha=1.0, label=labelx)
        _plt.plot(self._twiss0.spos, bbeaty,
                  color='r', alpha=0.8, label=labely)
        _plt.xlabel('spos [m]')
        _plt.ylabel('Beta Beating [%]')
        _plt.title('Beta Beating')
        _plt.suptitle(self.id_famname +
                      '- Symmetrized optics and uncorrected tunes')
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        _plt.savefig(fpath + 'opt{}-ids-symm'.format(label1), dpi=300)

        # Compare optics between nominal value and all corrected
        results = self._calc_dtune_betabeat(twiss1=self._twiss3)
        dtunex, dtuney = results[0], results[1]
        bbeatx, bbeaty = results[2], results[3]
        bbeatx_rms, bbeaty_rms = results[4], results[5]
        bbeatx_absmax, bbeaty_absmax = results[6], results[7]
        print('symmetrized optics and corrected tunes:')
        print(f'dtunex: {dtunex:+.0e}')
        print(f'dtuney: {dtuney:+.0e}')
        print(
            f'bbetax: {bbeatx_rms:04.3f} % rms,' +
            f' {bbeatx_absmax:04.3f} % absmax')
        print(
            f'bbetay: {bbeaty_rms:04.3f} % rms,' +
            f' {bbeaty_absmax:04.3f} % absmax')

        _plt.figure(3)
        labelx = f'X ({bbeatx_rms:.3f} % rms)'
        labely = f'Y ({bbeaty_rms:.3f} % rms)'
        _plt.plot(self._twiss0.spos, bbeatx,
                  color='b', alpha=1.0, label=labelx)
        _plt.plot(self._twiss0.spos, bbeaty,
                  color='r', alpha=0.8, label=labely)
        _plt.xlabel('spos [m]')
        _plt.ylabel('Beta Beating [%]')
        _plt.title('Beta Beating' + '\n' + self._stg)
        _plt.suptitle(self.id_famname +
                      '- Symmetrized optics and corrected tunes')
        _plt.legend()
        _plt.grid()
        _plt.tight_layout()
        _plt.savefig(fpath + 'opt{}-ids-symm-tunes'.format(label1), dpi=300)
        _plt.show()
        _plt.clf()

    def _correct_beta(self, straight_nr, knobs, goal_beta, goal_alpha):
        dk_tot = _np.zeros(len(knobs))
        for i in range(7):
            dk = optics.correct_symmetry_withbeta(
                self._model_id, straight_nr, goal_beta, goal_alpha)
            print('iteration #{}, dK: {}'.format(i+1, dk))
            dk_tot += dk
        self._stg = str()
        for i, fam in enumerate(knobs):
            self._stg += '{:<9s} dK: {:+9.4f} 1/m² \n'.format(fam, dk_tot[i])
        print(self._stg)
        self._twiss2, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices='closed')
        print()

    def _correct_tunes(self, goal_tunes):
        tunes = self._twiss1.mux[-1]/_np.pi/2, self._twiss1.muy[-1]/_np.pi/2
        print('init    tunes: {:.9f} {:.9f}'.format(tunes[0], tunes[1]))
        for i in range(2):
            optics.correct_tunes_twoknobs(self._model_id, goal_tunes)
            twiss, *_ = pyaccel.optics.calc_twiss(self._model_id)
            tunes = twiss.mux[-1]/_np.pi/2, twiss.muy[-1]/_np.pi/2
            print('iter #{} tunes: {:.9f} {:.9f}'.format(
                i+1, tunes[0], tunes[1]))
        print('goal    tunes: {:.9f} {:.9f}'.format(
            goal_tunes[0], goal_tunes[1]))
        self._twiss3, *_ = pyaccel.optics.calc_twiss(
            self._model_id, indices='closed')
        print()

    def _correct_optics(self, width, phase, gap):

        # create unperturbed model for reference
        model0 = self._create_model_nominal()
        self._twiss0, *_ = pyaccel.optics.calc_twiss(model0, indices='closed')

        # create model with ID
        fname = self._get_kmap_filename(width=width, phase=phase, gap=gap,
                                        shift_flag=self.shift_flag,
                                        filter_flag=self.filter_flag)
        self._model_id, self._ids = self._create_model_ids(fname=fname)
        knobs, locs_beta, straight_nr = self._get_knobs_locs()

        print('element indices for straight section begin and end:')
        for idsubsec, locs_beta_ in locs_beta.items():
            print(idsubsec, locs_beta_)

        print('local quadrupole fams: ')
        for idsubsec, knobs_ in knobs.items():
            print(idsubsec, knobs_)

        # correct orbit
        kicks, spos_bpms, codx_c, cody_c, codx_u, cody_u, bpms = \
            orbcorr.correct_orbit_fb(
                model0, self._model_id, corr_system=self.corr_system,
                nr_steps=1, plot_flag=self.orbcorr_plot_flag)

        # calculate beta beating and delta tunes
        self._analysis_uncorrected_perturbation(plot_flag=False)

        # get list of ID model indices and set rescale_kicks to zero
        ids_ind_all = orbcorr.get_ids_indices(self._model_id)
        rescale_kicks_orig = list()
        for idx in range(len(ids_ind_all)//2):
            ind_id = ids_ind_all[2*idx:2*(idx+1)]
            rescale_kicks_orig.append(self._model_id[ind_id[0]].rescale_kicks)
            self._model_id[ind_id[0]].rescale_kicks = 0
            self._model_id[ind_id[1]].rescale_kicks = 0

        # loop over IDs turning rescale_kicks on, one by one.
        for idx in range(len(ids_ind_all)//2):

            # turn rescale_kicks on for ID index idx
            ind_id = ids_ind_all[2*idx:2*(idx+1)]
            self._model_id[ind_id[0]].rescale_kicks = rescale_kicks_orig[idx]
            self._model_id[ind_id[1]].rescale_kicks = rescale_kicks_orig[idx]
            fam_name = self._model_id[ind_id[0]].fam_name
            # print(idx, ind_id)
            # continue

            # search knob and straight_nr for ID index idx
            for subsec in knobs:
                straight_nr_ = straight_nr[subsec]
                knobs_ = knobs[subsec]
                locs_beta_ = locs_beta[subsec]
                if min(locs_beta_) < ind_id[0] and ind_id[1] < max(locs_beta_):
                    break

            k = self._calc_coupling(x0=1e-6, nturns=1000)
            print()
            print('symmetrizing ID {} in subsec {}'.format(fam_name, subsec))

            # calculate nominal twiss
            goal_tunes = _np.array(
                [self._twiss0.mux[-1]/2/_np.pi, self._twiss0.muy[-1]/2/_np.pi])
            goal_beta = _np.array(
                [self._twiss0.betax[locs_beta_],
                 self._twiss0.betay[locs_beta_]])
            goal_alpha = _np.array(
                [self._twiss0.alphax[locs_beta_],
                 self._twiss0.alphay[locs_beta_]])
            print('goal_beta:')
            print(goal_beta)

            # symmetrize optics (local quad fam knobs)
            if self._beta_flag:
                self._correct_beta(
                    straight_nr_, knobs_, goal_beta, goal_alpha)
                # correct tunes
                self._correct_tunes(goal_tunes)

                if self.bb_plot_flag:
                    # plot results
                    self._plot_beta_beating(width=width, phase=phase, gap=gap)

        return self._model_id

    def _analysis_dynapt(self, model, width, phase, gap):

        model.radiation_on = 0
        model.cavity_on = False
        model.vchamber_on = True

        dynapxy = DynapXY(model)
        dynapxy.params.x_nrpts = 40
        dynapxy.params.y_nrpts = 20
        dynapxy.params.nrturns = 1*1024
        print(dynapxy)
        dynapxy.do_tracking()
        dynapxy.process_data()
        fig, *ax = dynapxy.make_figure_diffusion(orders=(1, 2, 3, 4),
                                                 nuy_bounds=(14.12, 14.45),
                                                 nux_bounds=(49.05, 49.50))
        fig.show()

        fpath = self._get_data_path(width=width, phase=phase, gap=gap)
        print(fpath)
        label1 = ['', '-ids-nonsymm', '-ids-symm'][self.calc_type]
        label2 = {False: '-nominal', True: '-fittedmodel'}[self.fitted_model]
        fig_name = fpath + 'dynapt{}{}.png'.format(label2, label1)
        fig.savefig(fig_name, dpi=300, format='png')
        _plt.figure(1)
        _plt.clf()
        _plt.figure(2)
        _plt.clf()

    def run_analysis_dynapt(self, width, phase, gap):
        if self.calc_type == self.CALC_TYPES.nominal:
            model = self._create_model_nominal()
        elif self.calc_type in (
                self.CALC_TYPES.symmetrized, self.CALC_TYPES.nonsymmetrized):
            self._beta_flag = self.calc_type == self.CALC_TYPES.symmetrized
            model = self._correct_optics(
                width=width, phase=phase, gap=gap)
        else:
            raise ValueError('Invalid calc_type')

        self._analysis_dynapt(model=model, width=width, phase=phase, gap=gap)
