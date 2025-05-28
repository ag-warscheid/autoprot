import hashlib
import os
from subprocess import run, STDOUT, Popen, PIPE
import pandas as pd

# this is a pointer to the module object instance itself.
module_pointer = __import__(__name__.split(".")[0])
config_dir = {}


def write_data_for_r(df, cols, write_csv=True, return_hash=False, tool=''):
    # Get a deterministic byte representation of the DataFrame
    hash_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    # Compute SHA256 and return first 10 characters of hex digest
    hash = hashlib.sha256(hash_bytes).hexdigest()[:10]

    d = os.getcwd()
    data_loc = d + f"/{hash}{tool}_input.csv"
    output_loc = d + f"/{hash}{tool}_output.csv"

    if not write_csv:
        if return_hash:
            return data_loc, output_loc, hash
        return data_loc, output_loc

    if "UID" not in df.columns:
        # UID is basically a row index starting at 1
        df["UID"] = range(1, df.shape[0] + 1)

    if not isinstance(cols, list):
        cols = cols.to_list()

    df[["UID"] + cols].to_csv(data_loc, sep="\t", index=False)
    if return_hash:
        return data_loc, output_loc, hash+tool
    return data_loc, output_loc


def check_r_install():
    base_path = str(os.path.join(os.path.dirname(os.path.realpath(__file__))))

    if not os.path.isfile(os.path.join(base_path, "autoprot.conf")):
        with open(os.path.join(base_path, "autoprot.conf"), "w") as wf:
            wf.write(
                f"R = PATH_TO_RSCRIPT\nRFUNCTIONS = {os.path.join(base_path, 'RFunctions.R')}"
            )
        raise OSError(
            "No R installation configured. Generated autoprot.conf. Please edit and try again."
        )
    else:
        with open(os.path.join(base_path, "autoprot.conf"), "r") as rf:
            for line in rf:
                split_line = [x.strip() for x in line.split("=")]
                if len(split_line) == 2:
                    k, v = [x.strip() for x in line.split("=")]
                    global config_dir  # we want to change config_dir in the global scope
                    config_dir[k] = v

    if "RSCRIPT" in config_dir.keys():
        print(
            "WARNING: The syntax of autoprot.conf has changed. Please adapt your paths accordingly."
        )

    if not os.path.isfile(config_dir["R"]):
        raise OSError(
            "The R variable should point to the Rscript executable. Make sure that it is not the R executable."
            "The currently configured path is: " + config_dir["R"]
        )

    if not os.path.isfile(config_dir["RFUNCTIONS"]):
        raise OSError(
            f"The RFUNCTIONS variable should point to the RFunctions.R file in your local autoprot "
            f'directory and not to {config_dir["RFUNCTIONS"]}'
        )

    if module_pointer.check_r_install is False:
        print("Checking R installation...")
        cmd = [
            config_dir["R"],
            "--vanilla",
            config_dir["RFUNCTIONS"],
            "functest",
            "",  # data location
            "",  # output file,
            "",  # kind of test
            "",  # design location
        ]

        # this enables real-time output of the R script
        with Popen(cmd, stdout=PIPE, stderr=STDOUT) as p:
            while True:
                output = p.stdout.readline().rstrip().decode("utf-8")
                if output == "" and p.poll() is not None:
                    break
                if output:
                    print(output.strip())

        # write out a description of the R environment
        write_description()
        # set the check_r_install to True to avoid running the R script again
        module_pointer.check_r_install = True
        print("R installation check complete.")


def write_description():
    """
    This functions writes a summary of the installed R packages to file.
    """
    run(
        [
            config_dir["R"],
            "-e",
            "write.csv(as.data.frame(installed.packages()), 'R_environment.csv', "
            "row.names = FALSE)",
        ],
        capture_output=False,
    )


def return_r_path():
    check_r_install()
    return config_dir["RFUNCTIONS"], config_dir["R"]


def run_r_command(command, print_r):
    p = run(command, capture_output=True, text=True, universal_newlines=True)
    if print_r:
        print(p.stdout)
        print(p.stderr)
