import os
from subprocess import check_output, run

config_dir = {}


def check_r_install():
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))

    if not os.path.isfile(os.path.join(base_path, 'autoprot.conf')):
        with open(os.path.join(base_path, 'autoprot.conf'), 'w') as wf:
            wf.write(f"R = PATH_TO_RSCRIPT\nRFUNCTIONS = {os.path.join(base_path, 'RFunctions.R')}")
        raise OSError('No R installation configured. Generated autoprot.conf. Please edit and try again.')
    else:
        with open(os.path.join(base_path, 'autoprot.conf'), 'r') as rf:
            for line in rf:
                split_line = [x.strip() for x in line.split('=')]
                if len(split_line) == 2:
                    k, v = [x.strip() for x in line.split('=')]
                    global config_dir  # we want to change config_dir in the global scope
                    config_dir[k] = v

    if 'RSCRIPT' in config_dir.keys():
        print('WARNING: The syntax of autoprot.conf has changed. Please adapt your paths accordingly.')

    if not os.path.isfile(config_dir['R']):
        raise OSError(
            'The R variable should point to the Rscript executable. Make sure that it is not the R executable.')

    if not os.path.isfile(config_dir['RFUNCTIONS']):
        raise OSError(f'The RFUNCTIONS variable should point to the RFunctions.R file in your local autoprot '
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

    write_description()


def write_description():
    """
    This functions writes a summary of the installed R packages to file.
    """
    p = run([config_dir['R'], "-e", "write.csv(as.data.frame(installed.packages()), 'R_environment.csv', "
                                    "row.names = FALSE)"],
            capture_output=False)


def return_r_path():
    check_r_install()
    return config_dir['RFUNCTIONS'], config_dir['R']


def run_r_command(command, print_r):
    p = run(command,
            capture_output=True,
            text=True,
            universal_newlines=True)
    if print_r:
        print(p.stdout)
        print(p.stderr)
