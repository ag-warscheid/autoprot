import os
from subprocess import check_output, STDOUT, run

# global var that stores the paths to the config files
global config_dir
# noinspection PyRedeclaration
config_dir = {}


def check_r_install():
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    if not os.path.isfile(os.path.join(base_path, 'autoprot.conf')):
        with open(os.path.join(base_path, 'autoprot.conf'), 'w') as wf:
            wf.write(f"R = PATH_TO_RSCRIPT\nRFUNCTIONS = {os.path.join(base_path, 'RFunctions.R')}")
        raise Exception('No R installation configured. Generated autoprot.conf. Please edit and try again.')
    else:
        with open(os.path.join(base_path, 'autoprot.conf'), 'r') as rf:
            for line in rf.readlines():
                splitline = [x.strip() for x in line.split('=')]
                if len(splitline) == 2:
                    k, v = [x.strip() for x in line.split('=')]
                    config_dir[k] = v

    if 'RSCRIPT' in config_dir.keys():
        print('WARNING: The syntax of autoprot.conf has changed. Please adapt your paths accordingly.')

    if not os.path.isfile(config_dir['R']):
        raise Exception(
            'The R variable should point to the Rscript executable. Make sure that it is not the R executable.')

    if not os.path.isfile(config_dir['RFUNCTIONS']):
        raise Exception(f'The FUNCTIONS variable should point to the RFunctions.R file in your local autoprot '
                        f'directory and not to {config_dir["RFUNCTIONS"]}')

    check_output([config_dir['R'],
                  '--vanilla',
                  config_dir['RFUNCTIONS'],
                  'functest',
                  '',  # data location
                  '',  # output file,
                  '',  # kind of test
                  ''  # design location
                  ])


def return_r_path():
    check_r_install()
    return config_dir['RSCRIPT'], config_dir['R']
