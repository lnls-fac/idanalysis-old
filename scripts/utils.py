from idanalysis import DeltaData


DEFAULT_RANDOM_IDS = True
FOLDER_BASE = '/home/ximenes/repos-dev/' 


def create_deltadata(random_ids=DEFAULT_RANDOM_IDS):
    if random_ids:
        folder = FOLDER_BASE + DeltaData.FOLDER_SABIA_MAPS_ERR
        configs = DeltaData.SABIA_CONFIGS_ERR
    else:
        folder = FOLDER_BASE + DeltaData.FOLDER_SABIA_MAPS
        configs = DeltaData.SABIA_CONFIGS
    deltadata = DeltaData(folder=folder, configs=configs)
    return deltadata


def get_config_names(data):
    names = []
    for config in data:
        names.append(config)
    return names
