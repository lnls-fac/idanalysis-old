#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize


from imaids import utils as ima_utils

from mathphys.functions import save_pickle
from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS
from utils import ID_PERIOD


DEF_RK_S_STEP = 0.2  # [mm]

_phase25n = ['ID4083', 'ID4103', 'ID4088', 'ID4093', 'ID4098']  # phase -25
_phase16n = ['ID4081', 'ID4101', 'ID4086', 'ID4091', 'ID4096']  # phase -16
_phase00p = ['ID4079', 'ID4099', 'ID4084', 'ID4089', 'ID4094']  # phase  00
_phase16p = ['ID4080', 'ID4100', 'ID4085', 'ID4090', 'ID4095']  # phase  16
_phase25p = ['ID4082', 'ID4102', 'ID4087', 'ID4092', 'ID4097']  # phase  25
GAPS = ['22.0', '23.3', '25.7', '29.3', '40.9']
PHASES = ['-25.00', '-16.39', '+00.00', '+16.39', '+25.00']
CONFIGS = [_phase25n, _phase16n, _phase00p, _phase16p, _phase25p]


def run_rk_traj(rk_s_step=DEF_RK_S_STEP):
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
    save_pickle(rk_traj_data, fpath + 'rk_traj_data.pickle')
    

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


def calc_eff_field(rz, field):
    """."""
    freqs = 2*np.pi*np.array([1/50, 3/50, 5/50])
    amps, *_ = ima_utils.fit_fourier_components(field, freqs, rz)
    return amps


def function_field_vs_gap(gap, a, b, c):
    """."""
    amp = a*np.exp(-b*gap) + c
    return amp


def plot_field_vs_gap(phase, bx, by, rz, tabulate_flag=True):
    """."""
    beffx_list, beffy_list = [], []
    keffx_list, keffy_list = [], []
    for gap in GAPS:
        rz_ = rz[gap]
        bx_, by_, = bx[gap], by[gap]
        fraction = int(len(rz_)/4)

        amps = calc_eff_field(rz_[fraction:3*fraction], bx_[fraction:3*fraction])
        beff = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keff = ima_utils.calc_deflection_parameter(beff, ID_PERIOD/1000)
        beffx_list.append(beff)
        keffx_list.append(keff)

        amps = calc_eff_field(rz_[fraction:3*fraction], by_[fraction:3*fraction])
        beffy = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keffy = ima_utils.calc_deflection_parameter(beffy, ID_PERIOD/1000)
        beffy_list.append(beffy)
        keffy_list.append(keffy)

    gap_array = np.array([float(gap) for gap in GAPS])
    gaps_fit = np.arange(22, 42, 1)
    fig, ax1 = plt.subplots()

    gap_array = gap_array/ID_PERIOD
    curve_fit = optimize.curve_fit(
        function_field_vs_gap, gap_array, beffx_list)[0]

    a, b, c = curve_fit[:3]
    ax, bx, cx = a, b, c
    label = ' Bx fit {:.2f}*exp(-{:.2f}*gap/λ) + {:.2f}'.format(a, b, c)
    gap_array *= ID_PERIOD
    fitted_curve = function_field_vs_gap(gaps_fit/ID_PERIOD, a, b, c)
    ax2 = ax1.twinx()
    ax1.plot(gaps_fit, fitted_curve, '--', color='b', label=label)
    ax1.plot(gap_array, beffx_list, 'o', color='b', label='Measurement')
    ax2.plot(gap_array, keffx_list, 'o', color='b')

    gap_array = gap_array/ID_PERIOD
    curve_fit = optimize.curve_fit(
        function_field_vs_gap, gap_array, beffy_list)[0]
    a, b, c = curve_fit[:3]
    ay, by, cy = a, b, c
    table_b_vs_gap(ax, bx, cx, ay, by, cy, phase, tabulate_flag)
    label = 'By fit {:.2f}*exp(-{:.2f}*gap/λ) + {:.2f}'.format(a, b, c)
    gap_array *= ID_PERIOD
    fitted_curve = function_field_vs_gap(gaps_fit/ID_PERIOD, a, b, c)
    ax1.plot(gaps_fit, fitted_curve, '--', color='C1', label=label)
    ax1.plot(gap_array, beffy_list, 'o', color='C1', label='Measurement')
    ax2.plot(gap_array, keffy_list, 'o', color='C1')

    ax1.set_xlabel('Gap [mm]')
    ax1.set_ylabel('Beff [T]')
    ax2.set_ylabel('Keff')
    ax1.legend()
    ax1.grid()
    title = 'Field for phase ' + phase + ' mm'
    plt.title(title)
    fig_path = 'results/phase-organized/' + phase + '/'
    plt.savefig(fig_path + 'field-amplitude-vs-gap.png', dpi=300)
    plt.close()


def table_b_vs_gap(ax, bx, cx, ay, by, cy, phase, tabulate_flag=True):
    """."""
    row = ['Gap [mm]', 'Bx [T]', 'By [T]']
    rows = []
    rows.append(row)
    gaps2 = np.linspace(20, 50, 30)
    beffx = function_field_vs_gap(gaps2/50, ax, bx, cx)
    beffy = function_field_vs_gap(gaps2/50, ay, by, cy)
    for i, gap in enumerate(gaps2):
        gapf = format(gap, '03.2f')
        bx = format(beffx[i], '03.2f')
        by = format(beffy[i], '03.2f')
        row = [gapf, bx, by]
        rows.append(row)

    if tabulate_flag:
        from tabulate import tabulate
        print('Tabulate Latex for phase {} mm: '.format(phase))
        print(tabulate(rows, headers='firstrow', tablefmt='latex'))


def run_multipoles(
        phase_config, idx, traj_init_rx, traj_init_ry, rk_s_step=0.2,
        tabulate_flag=True):
    """."""
    phase = PHASES[idx]
    rz_dict = dict()
    skew_quad_dict = dict()
    normal_quad_dict = dict()
    skew_sext_dict = dict()
    normal_sext_dict = dict()
    i_nquad_dict = dict()
    i_squad_dict = dict()
    i_nsext_dict = dict()
    i_ssext_dict = dict()

    for i, idconfig in enumerate(phase_config):
        gap = GAPS[i]
        # create IDKickMap and calc trajectory
        idkickmap = create_idkickmap(idconfig)
        idkickmap.rk_s_step = rk_s_step
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)
        traj = idkickmap.traj
        rz = traj.rz
        rz_dict[gap] = rz

        # multipolar analysis
        n_list = np.arange(0, 3, 1).tolist()
        s_list = np.arange(0, 3, 1).tolist()
        idkickmap.fmap_config.multipoles_perpendicular_grid = np.linspace(-3, 3, 7)
        idkickmap.fmap_config.multipoles_normal_field_fitting_monomials = n_list
        idkickmap.fmap_config.multipoles_skew_field_fitting_monomials = s_list
        idkickmap.fmap_config.multipoles_r0 = 12  # [mm]
        idkickmap.fmap_config.normalization_monomial = 0
        IDKickMap.multipoles_analysis(idkickmap.fmap_config)
        multipoles = idkickmap.fmap_config.multipoles
        normal_quad = multipoles.normal_multipoles[1, :]
        skew_quad = multipoles.skew_multipoles[1, :]
        normal_sext = multipoles.normal_multipoles[2, :]
        skew_sext = multipoles.skew_multipoles[2, :]
        i_nquad = multipoles.normal_multipoles_integral[1]
        i_squad = multipoles.skew_multipoles_integral[1]
        i_nsext = multipoles.normal_multipoles_integral[2]
        i_ssext = multipoles.skew_multipoles_integral[2]
        normal_quad_dict[gap] = normal_quad
        skew_quad_dict[gap] = skew_quad
        normal_sext_dict[gap] = normal_sext
        skew_sext_dict[gap] = skew_sext
        i_nquad_dict[gap] = i_nquad
        i_squad_dict[gap] = i_squad
        i_nsext_dict[gap] = i_nsext
        i_ssext_dict[gap] = i_ssext

    plt.figure(1)
    colors = ['b', 'g', 'C1', 'r', 'k']
    for i, idconfig in enumerate(phase_config):
        gap = GAPS[i]
        nquad = normal_quad_dict[gap]
        plt.plot(rz, nquad, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Quadrupolar component [T/m]')
    plt.grid()
    plt.legend()
    plt.title(
        'Normal quadrupole for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/Normal quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(2)
    for i, idconfig in enumerate(phase_config):
        gap = GAPS[i]
        squad = skew_quad_dict[gap]
        plt.plot(rz, squad, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Quadrupolar component [T/m]')
    plt.grid()
    plt.legend()
    plt.title(
        'Skew quadrupole for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/Skew quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(3)
    for i, idconfig in enumerate(phase_config):
        gap = GAPS[i]
        nsext = normal_sext_dict[gap]
        plt.plot(rz, nsext, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(
        'Normal sextupole for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/Normal sextupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(4)
    for i, idconfig in enumerate(phase_config):
        gap = GAPS[i]
        ssext = skew_sext_dict[gap]
        plt.plot(rz, ssext, colors[i], label=GAPS[i] + ' mm')
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(
        'Skew sextupole for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/Skew sextupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # generate table
    row1 = [
        'Gap [mm]',
        'Normal quadrupole [T]',
        'Skew quadrupole [T]',
        'Normal sextupole [T/m]',
        'Skew sextupole [T/m]']

    row_list = []
    row_list.append(row1)
    for gap in GAPS:
        i_nquadf = format(i_nquad_dict[gap], '+5.4f')
        i_squadf = format(i_squad_dict[gap], '+5.4f')
        i_nsextf = format(i_nsext_dict[gap], '+5.4f')
        i_ssextf = format(i_ssext_dict[gap], '+5.4f')
        row = [
            gap,
            i_nquadf,
            i_squadf,
            i_nsextf,
            i_ssextf]
        row_list.append(row)

    if tabulate_flag:
        from tabulate import tabulate
        # print('Tabulate Table for phase {} mm: '.format(phase))
        # print(tabulate(row_list, headers='firstrow'))

        print('Tabulate Latex for phase {} mm: '.format(phase))
        print(tabulate(row_list, headers='firstrow', tablefmt='latex'))

    return


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


def plot_rk_traj(traj_data, phase, configs, tabulate_flag=True):
    """."""
    data = traj_data
    bx, by, bz = data['bx'], data['by'], data['bz']
    rx, ry, rz = data['rx'], data['ry'], data['rz']
    fmapbx, fmapby = data['fmapbx'], data['fmapby']
    fmaprz = data['fmaprz']
    px, py = data['px'], data['py']
    i1bx, i1by = data['i1bx'], data['i1by']
    i2bx, i2by = data['i2bx'], data['i2by']

    colors = ['b', 'g', 'C1', 'r', 'k']
    fig_path = 'results/phasr-organized/' + phase + '/'
    title_fld_sufix = ' for phase {} mm'.format(phase)
    dpi = 300

    # plot bx
    plt.figure(1)
    for i in range(len(configs)):
        gap = GAPS[i]
        rz_, bx_ = rz[gap], bx[gap]
        label = GAPS[i] + ' mm'
        plt.plot(rz_, bx_/np.max(np.abs(bx_)), colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('Normalized field')
    plt.grid()
    plt.legend()
    plt.title('Horizontal field' + title_fld_sufix)
    plt.savefig(fig_path + 'field-bx.png', dpi=dpi)
    plt.close()

    # plot by
    plt.figure(2)
    for i in range(len(configs)):
        gap = GAPS[i]
        rz_, by_ = rz[gap], by[gap]
        label = GAPS[i] + ' mm'
        plt.plot(rz_, by_/np.max(np.abs(by_)), colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('Normalized field')
    plt.grid()
    plt.legend()
    plt.title('Vertical field' + title_fld_sufix)
    plt.savefig(fig_path + 'field-by.png', dpi=dpi)
    plt.close()

    # plot bz
    plt.figure(3)
    for i in range(len(configs)):
        gap = GAPS[i]
        rz_, bz_ = rz[gap], bz[gap]
        label = GAPS[i] + ' mm'
        plt.plot(rz_, bz_/np.max(np.abs(bz_)), colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('Normalized field')
    plt.grid()
    plt.legend()
    plt.title('Longitudinal field' + title_fld_sufix)
    plt.savefig(fig_path + 'field-bz.png', dpi=dpi)
    plt.close()

    # plot rx
    plt.figure(4)
    for i in range(len(configs)):
        gap = GAPS[i]
        rz_, rx_ = rz[gap], 1e3*rx[gap]
        label = GAPS[i] + ' mm' + ' rx @ end: {:+.2f} um'.format(rx_[-1])
        plt.plot(rz_, rx_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Pos for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/'
    plt.savefig(fig_path + 'traj-posx.png', dpi=dpi)
    plt.close()

    # plot px
    plt.figure(5)
    for i in range(len(configs)):
        gap = GAPS[i]
        rz_, px_ = rz[gap], 1e6*px[gap]
        label = GAPS[i] + ' mm' + ' px @ end: {:+.2f} urad'.format(px_[-1])
        plt.plot(rz_, px_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Ang for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/'
    plt.savefig(fig_path + 'traj-angx.png', dpi=dpi)
    plt.close()

    # plot ry
    plt.figure(6)
    for i in range(len(configs)):
        gap = GAPS[i]
        rz_, ry_ = rz[gap], 1e3*ry[gap]
        label = GAPS[i] + ' mm' + ' ry @ end: {:+.2f} um'.format(ry_[-1])
        plt.plot(rz_, ry_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Pos for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/'
    plt.savefig(fig_path + 'traj-posy.png', dpi=dpi)
    plt.close()

    # plot py
    plt.figure(7)
    for i in range(len(configs)):
        gap = GAPS[i]
        rz_, py_ = rz[gap], 1e6*py[gap]
        label = GAPS[i] + ' mm' + ' py @ end: {:+.2f} urad'.format(py_[-1])
        plt.plot(rz_, py_, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Ang for phase {} mm'.format(phase))
    fig_path = 'results/phasr-organized/' + phase + '/'
    plt.savefig(fig_path + 'traj-angy.png', dpi=dpi)
    plt.close()

    # plot field amplitude versus gap
    plot_field_vs_gap(phase, fmapbx, fmapby, fmaprz, tabulate_flag)

    # generate table
    row1 = [
        'Gap [mm]',
        'Bx 1st integral [G cm] / Δpy [urad]',
        'Bx 2nd integral [G cm²] / Δy [um]',
        'By 1st integral [G cm] / Δpx [urad]',
        'By 2nd integral [G cm²] / Δx [um]']
    row_list = []
    row_list.append(row1)
    for gap in GAPS:
        px_ = 1e6*px[gap][-1]
        py_ = 1e6*py[gap][-1]
        rx_ = 1e3*rx[gap][-1]
        ry_ = 1e3*ry[gap][-1]
        i1bx_ = 1e6*i1bx[gap][-1]
        i1by_ = 1e6*i1by[gap][-1]
        i2bx_ = 1e8*i2bx[gap][-1]
        i2by_ = 1e8*i2by[gap][-1]
        px_ = format(px_, '+4.2f')
        py_ = format(py_, '+4.2f')
        rx_ = format(rx_, '+4.2f')
        ry_ = format(ry_, '+4.2f')
        i1bx_ = format(i1bx_, '+5.1f')
        i1by_ = format(i1by_, '+5.1f')
        i2bx_ = format(i2bx_, '+3.2e')
        i2by_ = format(i2by_, '+3.2e')
        row = [
            gap,
            '{} / {}'.format(i1bx_, py_),
            '{} / {}'.format(i2bx_, ry_),
            '{} / {}'.format(i1by_, px_),
            '{} / {}'.format(i2by_, rx_)]
        row_list.append(row)

    if tabulate_flag:
        from tabulate import tabulate
        # print('Tabulate Table for phase {} mm: '.format(phase))
        # print(tabulate(row_list, headers='firstrow'))

        print('Tabulate Latex for phase {} mm: '.format(phase))
        print(tabulate(row_list, headers='firstrow', tablefmt='latex'))


def run_multipole_analysis(rk_s_step=DEF_RK_S_STEP, tabulate_flag=True):
    """."""
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]

    for i, phase in enumerate(CONFIGS):
        run_multipoles(
            float(PHASES[i]), i, traj_init_rx, traj_init_ry, rk_s_step=rk_s_step,
            tabulate_flag=tabulate_flag)


if __name__ == "__main__":
    """."""
    rk_s_step = 10 * DEF_RK_S_STEP
    # run_multipole_analysis()
    run_rk_traj(rk_s_step, tabulate_flag=False)
