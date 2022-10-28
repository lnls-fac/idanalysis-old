#!/usr/bin/env python-sirius

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as optimize
from tabulate import tabulate

from fieldmaptrack.common_analysis import multipoles_analysis
from idanalysis import IDKickMap
from idanalysis.fmap import FieldmapOnAxisAnalysis

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS

from imaids import utils as ima_utils


def fit_function(x, a, b, c):
    """."""
    return a*np.exp(b*-1*x) + c


def fit_measurement(gap, beff):
    """."""
    a0, b0, c0 = 3, 0.5, 0.5
    return optimize.curve_fit(
        fit_function, gap, beff)[0]


def calc_beff(z, B):
    """."""
    freqs = 2*np.pi*np.array([1/50, 3/50, 5/50])
    amps, *_ = ima_utils.fit_fourier_components(B, freqs, z)
    return amps


def table_b_vs_gap(ax, bx, cx, ay, by, cy, phase):
    """."""
    row = ['Gap [mm]', 'Bx [T]', 'By [T]']
    rows = []
    rows.append(row)
    gaps2 = np.linspace(20, 50, 30)
    beffx = fit_function(gaps2/50, ax, bx, cx)
    beffy = fit_function(gaps2/50, ay, by, cy)
    for i, gap in enumerate(gaps2):
        gapf = format(gap, '03.2f')
        bx = format(beffx[i], '03.2f')
        by = format(beffy[i], '03.2f')
        row = [gapf, bx, by]
        rows.append(row)

    print('Tabulate Latex for phase {} mm: '.format(phase))
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))


def plot_b_gap(phase_config, bx_dict, by_dict, rz_dict):
    """."""
    gap_list = []
    beff_list = []
    keff_list = []
    beffy_list = []
    keffy_list = []
    period = 50
    config = ID_CONFIGS[phase_config[0]]
    idx = config.find('fase')
    if config[idx+4] == '-':
        phase = config[idx+4:idx+10]
    else:
        phase = config[idx+4:idx+9]
    for i, idconfig in enumerate(phase_config):
        config = ID_CONFIGS[idconfig]
        idx = config.find('gap')
        gap = config[idx+4:idx+8]
        gap_list.append(float(gap))
        bx = bx_dict[gap]
        by = by_dict[gap]
        rz = rz_dict[gap]
        fraction = int(len(rz)/4)
        amps = calc_beff(rz[fraction:3*fraction], bx[fraction:3*fraction])
        beff = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keff = ima_utils.calc_deflection_parameter(beff, period/1000)
        beff_list.append(beff)
        keff_list.append(keff)

        amps = calc_beff(rz[fraction:3*fraction], by[fraction:3*fraction])
        beffy = np.sqrt(amps[0]**2+amps[1]**2+amps[2]**2)
        keffy = ima_utils.calc_deflection_parameter(beffy, period/1000)
        beffy_list.append(beffy)
        keffy_list.append(keffy)

    gap_array = np.array(gap_list)
    gaps = np.arange(22, 42, 1)
    fig, ax1 = plt.subplots()

    gap_array = gap_array/period
    curve_fit = fit_measurement(gap_array, beff_list)
    a = curve_fit[0]
    b = curve_fit[1]
    c = curve_fit[2]
    ax = a
    bx = b
    cx = c
    label = ' Bx fit {:.2f}*exp(-{:.2f}*gap/λ) + {:.2f}'.format(a, b, c)
    gap_array *= period
    fitted_curve = fit_function(gaps/period, a, b, c)
    ax2 = ax1.twinx()
    ax1.plot(gaps, fitted_curve, '--', color='b', label=label)
    ax1.plot(gap_array, beff_list, 'o', color='b', label='Measurement')
    ax2.plot(gap_array, keff_list, 'o', color='b')

    gap_array = gap_array/period
    curve_fit = fit_measurement(gap_array, beffy_list)
    a = curve_fit[0]
    b = curve_fit[1]
    c = curve_fit[2]
    ay = a
    by = b
    cy = c
    table_b_vs_gap(ax, bx, cx, ay, by, cy, phase)
    label = 'By fit {:.2f}*exp(-{:.2f}*gap/λ) + {:.2f}'.format(a, b, c)
    gap_array *= period
    fitted_curve = fit_function(gaps/period, a, b, c)
    ax1.plot(gaps, fitted_curve, '--', color='C1', label=label)
    ax1.plot(gap_array, beffy_list, 'o', color='C1', label='Measurement')
    ax2.plot(gap_array, keffy_list, 'o', color='C1')

    ax1.set_xlabel('Gap [mm]')
    ax1.set_ylabel('Beff [T]')
    ax2.set_ylabel('Keff')
    ax1.legend()
    ax1.grid()
    title = 'Field for phase ' + phase + ' mm'
    figname = '/B vs gap.png'
    plt.title(title)
    fig_path = 'results/phase organized/' + phase + figname
    plt.savefig(fig_path, dpi=300)
    plt.close()


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


def run_multipoles(
        phase_config, idx, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    gaps = ['22.0', '23.3', '25.7', '29.3', '40.9']
    phases = ['-25.00', '-16.39', '00.00', '16.39', '25.00']
    phase = phases[idx]
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
        gap = gaps[i]
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
    labels = ['22.0 mm', '23.3 mm', '25.7 mm', '29.3 mm', '40.9 mm']
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        nquad = normal_quad_dict[gap]
        plt.plot(rz, nquad, colors[i], label=labels[i])
    plt.xlabel('rz [mm]')
    plt.ylabel('Quadrupolar component [T/m]')
    plt.grid()
    plt.legend()
    plt.title(
        'Normal quadrupole for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/Normal quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(2)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        squad = skew_quad_dict[gap]
        plt.plot(rz, squad, colors[i], label=labels[i])
    plt.xlabel('rz [mm]')
    plt.ylabel('Quadrupolar component [T/m]')
    plt.grid()
    plt.legend()
    plt.title(
        'Skew quadrupole for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/Skew quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(3)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        nsext = normal_sext_dict[gap]
        plt.plot(rz, nsext, colors[i], label=labels[i])
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(
        'Normal sextupole for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/Normal sextupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(4)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        ssext = skew_sext_dict[gap]
        plt.plot(rz, ssext, colors[i], label=labels[i])
    plt.xlabel('rz [mm]')
    plt.ylabel('Sextupolar component [T/m²]')
    plt.grid()
    plt.legend()
    plt.title(
        'Skew sextupole for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/Skew sextupole.png'
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
    for gap in gaps:
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

    # print('Tabulate Table for phase {} mm: '.format(phase))
    # print(tabulate(row_list, headers='firstrow'))

    print('Tabulate Latex for phase {} mm: '.format(phase))
    print(tabulate(row_list, headers='firstrow', tablefmt='latex'))

    return


def plot_rk_traj(
        phase_config, idx, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    period = 50
    fieldtools = FieldmapOnAxisAnalysis()
    gaps = ['22.0', '23.3', '25.7', '29.3', '40.9']
    phases = ['-25.00', '-16.39', '00.00', '16.39', '25.00']
    phase = phases[idx]
    ibx_dict = dict()
    iibx_dict = dict()
    iby_dict = dict()
    iiby_dict = dict()
    rz_dict = dict()
    rx_dict = dict()
    ry_dict = dict()
    px_dict = dict()
    py_dict = dict()
    bx_dict = dict()
    by_dict = dict()
    bz_dict = dict()
    fmapbx_dict = dict()
    fmapby_dict = dict()
    fmaprz_dict = dict()

    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        # create IDKickMap and calc trajectory
        idkickmap = create_idkickmap(idconfig)
        idkickmap.rk_s_step = rk_s_step
        idkickmap.fmap_calc_trajectory(
            traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)
        traj = idkickmap.traj
        fmap = idkickmap.fmap_config.fmap

        rz = traj.rz
        rx = traj.rx
        ry = traj.ry
        px = traj.px
        py = traj.py
        bx = traj.bx
        by = traj.by
        bz = traj.bz

        fmaprz = fmap.rz
        fmapbx = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
        fmapby = fmap.by[fmap.ry_zero][fmap.rx_zero][:]

        fmaprz_dict[gap] = fmaprz
        fmapbx_dict[gap] = fmapbx
        fmapby_dict[gap] = fmapby

        rz_dict[gap] = rz
        rx_dict[gap] = rx
        ry_dict[gap] = ry
        px_dict[gap] = px
        py_dict[gap] = py
        bx_dict[gap] = bx
        by_dict[gap] = by
        bz_dict[gap] = bz

        ibx = fieldtools.calc_first_integral(bx, rz)
        iby = fieldtools.calc_first_integral(by, rz)
        ibx_dict[gap] = ibx
        iby_dict[gap] = iby
        iibx_dict[gap] = fieldtools.calc_second_integral(ibx, rz)
        iiby_dict[gap] = fieldtools.calc_second_integral(iby, rz)

    plt.figure(1)
    colors = ['b', 'g', 'C1', 'r', 'k']
    labels = ['22.0 mm', '23.3 mm', '25.7 mm', '29.3 mm', '40.9 mm']
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        bx = bx_dict[gap]
        plt.plot(rz, bx/np.max(np.abs(bx)), colors[i], label=labels[i])
    plt.xlabel('rz [mm]')
    plt.ylabel('Normalized field')
    plt.grid()
    plt.legend()
    plt.title(
        'Horizontal field for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/Bx.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(2)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        bx = by_dict[gap]
        plt.plot(rz, by/np.max(np.abs(by)), colors[i], label=labels[i])
    plt.xlabel('rz [mm]')
    plt.ylabel('Normalized field')
    plt.grid()
    plt.legend()
    plt.title(
        'Vertical field for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/By.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(3)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        bz = bz_dict[gap]
        plt.plot(rz, bz/np.max(np.abs(bz)), colors[i], label=labels[i])
    plt.xlabel('rz [mm]')
    plt.ylabel('Normalized field')
    plt.grid()
    plt.legend()
    plt.title(
        'Longitudinal field for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/Bz.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(4)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        rx = 1e3*rx_dict[gap]
        label = labels[i] + ' rx @ end: {:+.2f} um'.format(rx[-1])
        plt.plot(rz, rx, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Pos for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/RK Posx.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(5)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        px = 1e6*px_dict[gap]
        label = labels[i] + ' px @ end: {:+.2f} urad'.format(px[-1])
        plt.plot(rz, px, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Ang for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/RK Angx.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(6)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        ry = 1e3*ry_dict[gap]
        label = labels[i] + ' ry @ end: {:+.2f} um'.format(ry[-1])
        plt.plot(rz, ry, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Pos for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/RK Posy.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(7)
    for i, idconfig in enumerate(phase_config):
        gap = gaps[i]
        py = 1e6*py_dict[gap]
        label = labels[i] + ' py @ end: {:+.2f} urad'.format(py[-1])
        plt.plot(rz, py, colors[i], label=label)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Ang for phase {} mm'.format(phase))
    fig_path = 'results/phase organized/' + phase + '/RK Angy.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plot_b_gap(
        phase_config, fmapbx_dict, fmapby_dict, fmaprz_dict
    )

    # generate table
    row1 = [
        'Gap [mm]',
        'Bx 1st integral [G cm] / Δpy [urad]',
        'Bx 2nd integral [G cm²] / Δy [um]',
        'By 1st integral [G cm] / Δpx [urad]',
        'By 2nd integral [G cm²] / Δx [um]']
    row_list = []
    row_list.append(row1)
    for gap in gaps:
        px = 1e6*px_dict[gap][-1]
        py = 1e6*py_dict[gap][-1]
        rx = 1e3*rx_dict[gap][-1]
        ry = 1e3*ry_dict[gap][-1]
        ibx = 1e6*ibx_dict[gap][-1]
        iby = 1e6*iby_dict[gap][-1]
        iibx = 1e8*iibx_dict[gap][-1]
        iiby = 1e8*iiby_dict[gap][-1]
        px = format(px, '+4.2f')
        py = format(py, '+4.2f')
        rx = format(rx, '+4.2f')
        ry = format(ry, '+4.2f')
        ibx = format(ibx, '+5.1f')
        iby = format(iby, '+5.1f')
        iibx = format(iibx, '+3.2e')
        iiby = format(iiby, '+3.2e')
        row = [
            gap,
            '{} / {}'.format(ibx, py),
            '{} / {}'.format(iibx, ry),
            '{} / {}'.format(iby, px),
            '{} / {}'.format(iiby, rx)]
        row_list.append(row)

    # print('Tabulate Table for phase {} mm: '.format(phase))
    # print(tabulate(row_list, headers='firstrow'))

    print('Tabulate Latex for phase {} mm: '.format(phase))
    print(tabulate(row_list, headers='firstrow', tablefmt='latex'))

    return

    print('Field Integrals and traj for gap {} mm and phase {} mm'.format(
        gap, phase))
    print('IBx {:.3e} Tm'.format(ibx[-1]))
    print('py {:+.2f} urad'.format(1e6*traj.py[-1]))
    print()
    print('IIBx {:.3e} Tm²'.format(iibx[-1]))
    print('ry {:+.2f} um'.format(1e3*traj.ry[-1]))
    print()
    print('IBy {:.3e} Tm'.format(iby[-1]))
    print('px {:+.2f} urad'.format(1e6*traj.px[-1]))
    print()
    print('IIBy {:.3e} Tm²'.format(iiby[-1]))
    print('rx {:+.2f} um'.format(1e3*traj.rx[-1]))
    print()


def run_multipole_analysis():
    """."""
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]

    phase25n = ['ID4083', 'ID4103', 'ID4088', 'ID4093', 'ID4098']  # phase -25
    phase16n = ['ID4081', 'ID4101', 'ID4086', 'ID4091', 'ID4096']  # phase -16
    phase00 = ['ID4079', 'ID4099', 'ID4084', 'ID4089', 'ID4094']  # phase  00
    phase16 = ['ID4080', 'ID4100', 'ID4085', 'ID4090', 'ID4095']  # phase  16
    phase25 = ['ID4082', 'ID4102', 'ID4087', 'ID4092', 'ID4097']  # phase  25
    phase_list = [phase25n, phase16n, phase00, phase16, phase25]
    phases = [-25, -16.39, 0, 16.39, 25]
    for i, phase in enumerate(phase_list):
        run_multipoles(phase, i, traj_init_rx, traj_init_ry, rk_s_step=0.2)


def run_traj_analysis():
    """."""
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]

    phase25n = ['ID4083', 'ID4103', 'ID4088', 'ID4093', 'ID4098']  # phase -25
    phase16n = ['ID4081', 'ID4101', 'ID4086', 'ID4091', 'ID4096']  # phase -16
    phase00 = ['ID4079', 'ID4099', 'ID4084', 'ID4089', 'ID4094']  # phase  00
    phase16 = ['ID4080', 'ID4100', 'ID4085', 'ID4090', 'ID4095']  # phase  16
    phase25 = ['ID4082', 'ID4102', 'ID4087', 'ID4092', 'ID4097']  # phase  25
    phase_list = [phase25n, phase16n, phase00, phase16, phase25]
    phases = ['-25.00 mm', '-16.39 mm', '00.00 mm', '16.39 mm', '25.00 mm']
    gaps = [22.0, 23.3, 25.7, 29.3, 40.9]

    for i, phase in enumerate(phase_list):
        plot_rk_traj(phase, i, traj_init_rx, traj_init_ry, rk_s_step=0.2)


if __name__ == "__main__":
    """."""
    # run_multipole_analysis()
    run_traj_analysis()
