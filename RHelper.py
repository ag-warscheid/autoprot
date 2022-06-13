import os
from subprocess import check_output, STDOUT, run

def checkRinstall():
    
    basepath = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    
    if not os.path.isfile(os.path.join(basepath, 'autoprot.conf')):
        with open(os.path.join(basepath, 'autoprot.conf'),'w') as wf:
            wf.write("RSCRIPT = RSCRIPTPATH\nR = RPATH"
                     )
        raise Exception('No R installation configured. Generated autoprot.conf. Please edit and try again.')
    else:
        with open(os.path.join(basepath, 'autoprot.conf'),'r') as rf:
            
            global conf_dir
            
            conf_dir = dict()
            for line in rf.readlines():
                splitline = [x.strip() for x in line.split('=')]
                if len(splitline) == 2:
                    k, v = [x.strip() for x in line.split('=')]
                    conf_dir[k] = v

    if not os.path.isfile(conf_dir['RSCRIPT']):
        raise Exception('The RSCRIPT variable should point to the RFunctions.R file in your local autoprot directory and not to {}'.format(conf_dir['RSCRIPT']))
    elif not os.path.isfile(conf_dir['R']):
        raise Exception('The R variable should point to the Rscript executable. Make sure that it is not the R executable.')

    check_output([conf_dir['R'],
                  '--vanilla',
                  conf_dir['RSCRIPT'],
                  'functest',
                  '', #data location
                  '', #output file,
                  '', #kind of test
                  '' #design location
                  ], timeout=600)

def returnRPath():
    
    checkRinstall()
    return conf_dir['RSCRIPT'], conf_dir['R']