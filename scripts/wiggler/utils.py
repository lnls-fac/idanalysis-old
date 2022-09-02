"""."""

import numpy as np
import pyaccel
import pymodels

from idanalysis import IDKickMap

FOLDER_BASE = '/home/gabriel/repos-dev/'
#FOLDER_BASE = '/home/ximenes/repos-dev/'
DATA_PATH = '/wiggler-2T-STI/measurement/magnetic/hallprobe/'


WIGGLER_CONFIGS = {
    # wiggler without correctors
    'ID3971': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_199_50mm_Fieldmap_Z=-1650_1650mm_ID=3971.dat', 
    'ID3962': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_22_00mm_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3962.dat', 
    'ID3963': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_23_00mm_Fieldmap_Z=-1650_1650mm_ID=3963.dat', 
    'ID3964': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_23_95mm_Fieldmap_Z=-1650_1650mm_ID=3964.dat', 
    'ID3965': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_26_90mm_Fieldmap_Z=-1650_1650mm_ID=3965.dat', 
    'ID3972': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_300_00mm_Fieldmap_Z=-1650_1650mm_ID=3972.dat', 
    'ID3966': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_34_80mm_Fieldmap_Z=-1650_1650mm_ID=3966.dat', 
    'ID3967': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_42_70mm_Fieldmap_Z=-1650_1650mm_ID=3967.dat', 
    'ID3968': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_51_65mm_Fieldmap_Z=-1650_1650mm_ID=3968.dat', 
    'ID3969': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_59_60mm_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3969.dat', 
    'ID3970': 'wiggler_without_correctors/2022-08-22_Wiggler_STI_99_50mm_Fieldmap_Z=-1650_1650mm_ID=3970.dat', 
    
    # wiggler with correctors
    # gap 059.60mm
    'ID3979': 'gap 059.60mm/2022-08-25_WigglerSTI_059.60mm_U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3979.dat',
    'ID3984': 'gap 059.60mm/2022-08-25_WigglerSTI_059.60mm_U+0.64_D+0.87_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3984.dat',
    'ID3980': 'gap 059.60mm/2022-08-25_WigglerSTI_059.60mm_U+0.75_D-1.13_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3980.dat',
    'ID3981': 'gap 059.60mm/2022-08-25_WigglerSTI_059.60mm_U+5.00_D+5.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3981.dat',
    # gap 022.00mm
    'ID3977': 'gap 022.00mm/2022-08-25_WigglerSTI_022.00mm_U+0.00_D+0.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3977.dat',
    'ID3978': 'gap 022.00mm/2022-08-25_WigglerSTI_022.00mm_U-4.55_D+4.40_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=3978.dat',
    'ID4004': 'gap 022.00mm/2022-08-26_WigglerSTI_022.00mm_U+5.00_D+5.00_Fieldmap_X=-20_20mm_Z=-1650_1650mm_ID=4004.dat',
    # gap 023.00mm
    'ID3986': 'gap 023.00mm/2022-08-26_WigglerSTI_023.00mm_U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3986.dat',
    'ID4003': 'gap 023.00mm/2022-08-26_WigglerSTI_023.00mm_U-4.42_D+4.24_Fieldmap_Z=-1650_1650mm_ID=4003.dat',
    # gap 023.95mm
    'ID3987': 'gap 023.95mm/2022-08-26_WigglerSTI_023.95mm_U+0.00_D-0.00_Fieldmap_Z=-1650_1650mm_ID=3987.dat',
    'ID4002': 'gap 023.95mm/2022-08-26_WigglerSTI_023.95mm_U-4.15_D+4.29_Fieldmap_Z=-1650_1650mm_ID=4002.dat',
    # gap 026.90mm
    'ID3987': 'gap 026.90mm/2022-08-26_WigglerSTI_023.95mm_U+0.00_D-0.00_Fieldmap_Z=-1650_1650mm_ID=3987.dat',
    'ID4002': 'gap 026.90mm/2022-08-26_WigglerSTI_023.95mm_U-4.15_D+4.29_Fieldmap_Z=-1650_1650mm_ID=4002.dat',
    # gap 034.80mm
    'ID3989': 'gap 034.80mm/2022-08-26_WigglerSTI_034.80mm_U+0.00_D-0.00_Fieldmap_Z=-1650_1650mm_ID=3989.dat',
    'ID4000': 'gap 034.80mm/2022-08-26_WigglerSTI_034.80mm_U-2.73_D+2.35_Fieldmap_Z=-1650_1650mm_ID=4000.dat',
    # gap 042.70mm
    'ID3990': 'gap 042.70mm/2022-08-26_WigglerSTI_042.70mm_U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3990.dat',
    'ID3999': 'gap 042.70mm/2022-08-26_WigglerSTI_042.70mm_U-1.66_D+1.34_Fieldmap_Z=-1650_1650mm_ID=3999.dat',
    # gap 051.65mm
    'ID3991': 'gap 051.65mm/2022-08-26_WigglerSTI_051.65mm_U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3991.dat',
    'ID3998': 'gap 051.65mm/2022-08-26_WigglerSTI_051.65mm_U-0.32_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3998.dat',
    # gap 099.50mm
    'ID3992': 'gap 099.50mm/2022-08-26_WigglerSTI_099.50mm_U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3992.dat',
    'ID3997': 'gap 099.50mm/2022-08-26_WigglerSTI_099.50mm_U+0.37_D-0.64_Fieldmap_Z=-1650_1650mm_ID=3997.dat',
    # gap 199.50mm
    'ID3993': 'gap 199.50mm/2022-08-26_WigglerSTI_199.50mm_U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3993.dat',
    'ID3996': 'gap 199.50mm/2022-08-26_WigglerSTI_199.50mm_U-1.34_D+1.21_Fieldmap_Z=-1650_1650mm_ID=3996.dat',
    # gap 300.00mm
    'ID3994': 'gap 300.00mm/2022-08-26_WigglerSTI_300.00mm_U+0.00_D+0.00_Fieldmap_Z=-1650_1650mm_ID=3994.dat',
    'ID3995': 'gap 300.00mm/2022-08-26_WigglerSTI_300.00mm_U-0.66_D+0.41_Fieldmap_Z=-1650_1650mm_ID=3995.dat',
    }


def create_ids(nr_steps=None, rescale_kicks=None, rescale_length=None, config='ID3979'):
    # create IDs
    nr_steps = nr_steps or 40
    rescale_kicks = rescale_kicks if rescale_kicks is not None else 1.0
    rescale_length = \
        rescale_length if rescale_length is not None else 1
    fname = FOLDER_BASE + \
        'idanalysis/scripts/wiggler/results/'
    fname += 'kickmap-' + config + '.txt'
    print(fname)
    idkmap = IDKickMap(kmap_fname=fname)
    idkmap.load()
    kickx_up = idkmap.kickx_upstream  # [T².m²]
    kicky_up = idkmap.kicky_upstream  # [T².m²]
    kickx_down = idkmap.kickx_downstream  # [T².m²]
    kicky_down = idkmap.kicky_downstream  # [T².m²]
    termination_kicks = [kickx_up, kicky_up, kickx_down, kicky_down] 
    IDModel = pymodels.si.IDModel
    wig180 = IDModel(
        subsec = IDModel.SUBSECTIONS.ID14SB,
        file_name=fname,
        fam_name='WIG180', nr_steps=nr_steps,
        rescale_kicks=rescale_kicks, rescale_length=rescale_length,
        termination_kicks=termination_kicks)
    ids = [wig180, ]
    return ids


def get_orb4d(model):

    state_cavity = model.cavity_on
    state_radiation = model.radiation_on
    model.cavity_on = False
    # model.radiation_on = pyaccel.accelerator.RadiationStates.off

    # orbit
    closed_orbit = pyaccel.tracking.find_orbit4(model, indices='closed')
    codrx, codpx, codry, codpy = \
        closed_orbit[0], closed_orbit[1], closed_orbit[2], closed_orbit[3]

    model.cavity_on = state_cavity
    model.radiation_on = state_radiation

    return codrx, codpx, codry, codpy


def get_data_ID(fname):
    """."""
    _, idn =  fname.split('ID=')
    idn = idn.replace('.dat', '')
    return idn
