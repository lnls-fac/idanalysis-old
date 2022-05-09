#!/usr/bin/env python-sirius


from idanalysis import model

import utils
utils.FOLDER_BASE = '/home/ximenes/repos-dev/'
# utils.FOLDER_BASE = '/home/gabriel/repos-sirius/'

# create object with list of all possible EPU50 configurations
configs = utils.create_epudata()

kmap_fname = configs.get_kickmap_filename(configs[0])
ids = model.get_id_epu_list(kmap_fname, ids=None, nr_steps=40)

model0 = model.create_model(ids=None, vchamber_on=False)
model1 = model.create_model(ids=ids, vchamber_on=False)
print(model)

