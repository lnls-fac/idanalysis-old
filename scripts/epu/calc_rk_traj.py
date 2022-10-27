#!/usr/bin/env python-sirius

from fieldmaptrack.common_analysis import multipoles_analysis
from idanalysis.fmap import FieldmapOnAxisAnalysis
import numpy as np
import matplotlib.pyplot as plt

from idanalysis import IDKickMap

from utils import FOLDER_BASE
from utils import DATA_PATH
from utils import ID_CONFIGS


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


def multipolar_analysis(
        idconfig, gap, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    # create IDKickMap
    idkickmap = create_idkickmap(idconfig)

    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)
    traj = idkickmap.traj

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
    rz = traj.rz
    skew_quad = multipoles.skew_multipoles[1, :]
    normal_quad = multipoles.normal_multipoles[1, :]
    skew_sext = multipoles.skew_multipoles[2, :]
    normal_sext = multipoles.normal_multipoles[2, :]
    integral_nquad = np.trapz(normal_quad, rz/1000)
    integral_squad = np.trapz(skew_quad, rz/1000)
    integral_nsext = np.trapz(normal_sext, rz/1000)
    integral_ssext = np.trapz(skew_sext, rz/1000)
    i_nquad = multipoles.normal_multipoles_integral[1]
    i_squad = multipoles.skew_multipoles_integral[1]
    i_nsext = multipoles.normal_multipoles_integral[2]
    i_ssext = multipoles.skew_multipoles_integral[2]
    integral_multipole_list = [i_nquad, i_squad, i_nsext, i_ssext]
    plt.figure(1)
    plt.plot(rz, normal_quad, color='b')
    plt.grid()
    plt.xlabel('rz [mm]')
    plt.ylabel('Normal quadrupole [T/m]')
    plt.title('UVX EPU Normal Quadrupole Components')
    fig_path = 'results/gap ' + gap + '/' + idconfig + '/Nquad.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(2)
    plt.plot(rz, skew_quad, color='b')
    plt.grid()
    plt.xlabel('rz [mm]')
    plt.ylabel('Skew quadrupole [T/m]')
    plt.title('UVX EPU Skew Quadrupole Components')
    fig_path = 'results/gap ' + gap + '/' + idconfig + '/Squad.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(3)
    plt.plot(rz, normal_sext, color='r')
    plt.grid()
    plt.xlabel('rz [mm]')
    plt.ylabel('Normal sextupole [T/m²]')
    plt.title('UVX EPU Normal Sextupole Components')
    fig_path = 'results/gap ' + gap + '/' + idconfig + '/Nsext.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    plt.figure(4)
    plt.plot(rz, skew_sext, color='r')
    plt.grid()
    plt.xlabel('rz [mm]')
    plt.ylabel('Skew sextupole [T/m²]')
    plt.title('UVX EPU Skew Sextupole Components')
    fig_path = 'results/gap ' + gap + '/' + idconfig + '/Ssext.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()
    return integral_multipole_list


def plot_rk_traj(
        idconfig, gap, phase, traj_init_rx, traj_init_ry, rk_s_step=0.2):
    """."""
    # create IDKickMap
    idkickmap = create_idkickmap(idconfig)
    idkickmap.rk_s_step = rk_s_step
    idkickmap.fmap_calc_trajectory(
        traj_init_rx=traj_init_rx, traj_init_ry=traj_init_ry)

    traj = idkickmap.traj
    fmap = idkickmap.fmap_config.fmap
    rz = fmap.rz
    bx = fmap.bx[fmap.ry_zero][fmap.rx_zero][:]
    by = fmap.by[fmap.ry_zero][fmap.rx_zero][:]
    bz = fmap.bz[fmap.ry_zero][fmap.rx_zero][:]

    rz = traj.rz
    bx = traj.bx
    by = traj.by

    fieldtools = FieldmapOnAxisAnalysis()
    ibx = fieldtools.calc_first_integral(bx, rz)
    iby = fieldtools.calc_first_integral(by, rz)
    iibx = fieldtools.calc_second_integral(ibx, rz)
    iiby = fieldtools.calc_second_integral(iby, rz)

    plt.figure(1)
    plt.plot(rz, bx, color='b', label="Bx")
    plt.plot(rz, by, color='C1', label="By")
    # plt.plot(rz, bz, color='g', label="Bz")
    plt.xlabel('rz [mm]')
    plt.ylabel('Field [T]')
    plt.grid()
    plt.legend()
    plt.title(
        'Field components for gap {} mm and phase {} mm'.format(gap, phase))
    fig_path = 'results/gap ' + gap + '/' + idconfig + '/Field.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    labelx = 'rx @ end: {:+.1f} um'.format(1e3*traj.rx[-1])
    labely = 'ry @ end: {:+.1f} um'.format(1e3*traj.ry[-1])
    plt.figure(2)
    plt.plot(traj.rz, 1e3*traj.rx, '-', color='b', label=labelx)
    plt.plot(traj.rz, 1e3*traj.ry, '-', color='C1', label=labely)
    plt.xlabel('rz [mm]')
    plt.ylabel('pos [um]')
    plt.grid()
    plt.legend()
    plt.title(
        'RK Trajectory Pos for gap {} mm and phase {} mm'.format(gap, phase))
    fig_path = 'results/gap ' + gap + '/' + idconfig + '/RK Pos.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    labelx = 'px @ end: {:+.1f} urad'.format(1e6*traj.px[-1])
    labely = 'py @ end: {:+.1f} urad'.format(1e6*traj.py[-1])
    plt.figure(3)
    plt.plot(traj.rz, 1e6*traj.px, '-', color='b', label=labelx)
    plt.plot(traj.rz, 1e6*traj.py, '-', color='C1', label=labely)
    plt.xlabel('rz [mm]')
    plt.ylabel('ang [urad]')
    plt.legend()
    plt.grid()
    plt.title(
        'RK Trajectory Ang for gap {} mm and phase {} mm'.format(gap, phase))
    fig_path = 'results/gap ' + gap + '/' + idconfig + '/RK Ang.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print('Field Integrals and traj for gap {} mm and phase {} mm'.format(
        gap, phase))
    print('IBx {:.3e} Tm'.format(ibx[-1]))
    print('px {:+.2f} urad'.format(1e6*traj.py[-1]))
    print()
    print('IIBx {:.3e} Tm²'.format(iibx[-1]))
    print('rx {:+.2f} um'.format(1e3*traj.ry[-1]))
    print()
    print('IBy {:.3e} Tm'.format(iby[-1]))
    print('py {:+.2f} urad'.format(1e6*traj.px[-1]))
    print()
    print('IIBy {:.3e} Tm²'.format(iiby[-1]))
    print('ry {:+.2f} um'.format(1e3*traj.rx[-1]))
    print()


def run_multipole_analysis():
    """."""
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]

    gap22 = ['ID4083', 'ID4081', 'ID4079', 'ID4080', 'ID4082']  # gap 22.0
    gap23 = ['ID4103', 'ID4101', 'ID4099', 'ID4100', 'ID4102']  # gap 23.3
    gap25 = ['ID4088', 'ID4086', 'ID4084', 'ID4085', 'ID4087']  # gap 25.7
    gap29 = ['ID4093', 'ID4091', 'ID4089', 'ID4090', 'ID4092']  # gap 29.3
    gap40 = ['ID4098', 'ID4096', 'ID4094', 'ID4095', 'ID4097']  # gap 40.9
    gap_list = [gap22, gap23, gap25, gap29, gap40]
    phase = [-25, -16.39, 0, 16.39, 25]
    gaps_values = ['22.0', '23.3', '25.7', '29.3', '40.9']
    inormal_quad_dict = dict()
    iskew_quad_dict = dict()
    inormal_sext_dict = dict()
    iskew_sext_dict = dict()

    for i, gap in enumerate(gap_list):
        gap_value = gaps_values[i]
        inormal_quad = []
        iskew_quad = []
        inormal_sext = []
        iskew_sext = []
        for j, config in enumerate(gap):
            imultipoles = multipolar_analysis(
                config, gap_value, traj_init_rx, traj_init_ry, rk_s_step)
            inormal_quad.append(imultipoles[0])
            iskew_quad.append(imultipoles[1])
            inormal_sext.append(imultipoles[2])
            iskew_sext.append(imultipoles[3])
        inormal_quad_dict[gap_value] = inormal_quad
        iskew_quad_dict[gap_value] = iskew_quad
        inormal_sext_dict[gap_value] = inormal_sext
        iskew_sext_dict[gap_value] = iskew_sext

    # integrated normal quadrupole
    plt.figure(1)
    for i, gap in enumerate(gaps_values):
        plt.plot(phase, inormal_quad_dict[gap], 'o', label=gap)
    plt.title('UVX EPU Integrated Normal Quadrupole Components')
    plt.xlabel('Phase [mm]')
    plt.ylabel('Integrated quadrupole [T]')
    plt.grid()
    plt.legend()
    fig_path = 'results/Integrated normal quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # integrated skew quadrupole
    plt.figure(2)
    for i, gap in enumerate(gaps_values):
        plt.plot(phase, iskew_quad_dict[gap], 'o', label=gap)
    plt.title('UVX EPU Integrated Skew Quadrupole Components')
    plt.xlabel('Phase [mm]')
    plt.ylabel('Integrated quadrupole [T]')
    plt.grid()
    plt.legend()
    fig_path = 'results/Integrated skew quadrupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # integrated normal sextupole
    plt.figure(3)
    for i, gap in enumerate(gaps_values):
        plt.plot(phase, inormal_sext_dict[gap], 'o', label=gap)
    plt.title('UVX EPU Integrated Normal Sextupole Components')
    plt.xlabel('Phase [mm]')
    plt.ylabel('Integrated sextupole [T/m]')
    plt.grid()
    plt.legend()
    fig_path = 'results/Integrated normal sextupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # integrated skew sextupole
    plt.figure(4)
    for i, gap in enumerate(gaps_values):
        plt.plot(phase, iskew_sext_dict[gap], 'o', label=gap)
    plt.title('UVX EPU Integrated Skew Sextupole Components')
    plt.xlabel('Phase [mm]')
    plt.ylabel('Integrated sextupole [T/m]')
    plt.grid()
    plt.legend()
    fig_path = 'results/Integrated skew sextupole.png'
    plt.savefig(fig_path, dpi=300)
    plt.close()


def run_traj_analysis():
    """."""
    traj_init_rx = 0.0  # [mm]
    traj_init_ry = 0.0  # [mm]
    rk_s_step = 0.2  # [mm]

    gap22 = ['ID4083', 'ID4081', 'ID4079', 'ID4080', 'ID4082']  # gap 22.0
    gap23 = ['ID4103', 'ID4101', 'ID4099', 'ID4100', 'ID4102']  # gap 23.3
    gap25 = ['ID4088', 'ID4086', 'ID4084', 'ID4085', 'ID4087']  # gap 25.7
    gap29 = ['ID4093', 'ID4091', 'ID4089', 'ID4090', 'ID4092']  # gap 29.3
    gap40 = ['ID4098', 'ID4096', 'ID4094', 'ID4095', 'ID4097']  # gap 40.9
    gap_list = [gap22, gap23, gap25, gap29, gap40]
    phases = ['-25', '-16.39', '0', '16.39', '25']
    gaps_values = ['22.0', '23.3', '25.7', '29.3', '40.9']
    for i, idgap in enumerate(gap_list):
        gap = gaps_values[i]
        for j, config in enumerate(idgap):
            phase = phases[j]
            plot_rk_traj(
                config, gap, phase,  traj_init_rx, traj_init_ry, rk_s_step=0.2)


if __name__ == "__main__":
    """."""
    run_traj_analysis()
