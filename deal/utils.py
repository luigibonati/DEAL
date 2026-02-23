import numpy as np
from ase.data import covalent_radii as CR
from ase.data import chemical_symbols as CS
from ase.io import read
from ase import Atoms
import pandas as pd

import os

__all__ = [
    "average_along_cv",
    "compute_histogram",
    "plumed_to_pandas",
    "paletteFessa",
    "create_chemiscope_input",
]


def average_along_cv(value, cv, bins):
    h1, x = np.histogram(cv, bins=bins, weights=value)
    h2, _ = np.histogram(cv, bins=bins)

    mean = h1 / h2
    x = (x[:-1] + x[1:]) / 2

    return x, mean


def compute_histogram(value, bins, threshold=None):
    h1, x = np.histogram(value, bins=bins)
    x = (x[:-1] + x[1:]) / 2

    if threshold is not None:
        h2, _ = np.histogram(value[(value > threshold)], bins=bins)
        return x, (h1, h2)
    else:
        return x, h1


################################################################
## Functions to load COLVAR files
## source: https://mlcolvar.readthedocs.io/en/stable/_modules/mlcolvar/utils/io.html
################################################################


def is_plumed_file(filename):
    """
    Check if given file is in PLUMED format.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    bool
        wheter is a plumed output file
    """
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    is_plumed = True if " ".join(headers.columns[:2]) == "#! FIELDS" else False
    return is_plumed


def plumed_to_pandas(filename="./COLVAR"):
    """
    Load a PLUMED file and save it to a dataframe.

    Parameters
    ----------
    filename : string, optional
        PLUMED output file

    Returns
    -------
    df : DataFrame
        Collective variables dataframe
    """
    skip_rows = 1
    # Read header
    headers = pd.read_csv(filename, sep=" ", skipinitialspace=True, nrows=0)
    # Discard #! FIELDS
    headers = headers.columns[2:]
    # Load dataframe and use headers for columns names
    df = pd.read_csv(
        filename,
        sep=" ",
        skipinitialspace=True,
        header=None,
        skiprows=range(skip_rows),
        names=headers,
        comment="#",
    )

    return df


def load_dataframe(
    file_names, start=0, stop=None, stride=1, delete_download=True, **kwargs
):
    """Load dataframe(s) from file(s). It can be used also to open files from internet (if the string contains http).
    In case of PLUMED colvar files automatically handles the column names, otherwise it is just a wrapper for pd.load_csv function.

    Parameters
    ----------
    filenames : str or list[str]
        filenames to be loaded
    start: int, optional
        read from this row, default 0
    stop: int, optional
        read until this row, default None
    stride: int, optional
        read every this number, default 1
    delete_download: bool, optinal
        whether to delete the downloaded file after it has been loaded, default True.
    kwargs:
        keyword arguments passed to pd.load_csv function

    Returns
    -------
    pandas.DataFrame
        Dataframe

    Raises
    ------
    TypeError
        if data is not a valid type
    """

    # if it is a single string
    if type(file_names) == str:
        file_names = [file_names]
    elif type(file_names) != list:
        raise TypeError(
            f"only strings or list of strings are supported, not {type(file_names)}."
        )

    # list of file_names
    df_list = []
    for i, filename in enumerate(file_names):
        # check if filename is an url
        download = False
        if "http" in filename:
            download = True
            url = filename
            filename = "tmp_" + filename.split("/")[-1]
            urllib.request.urlretrieve(url, filename)

        # check if file is in PLUMED format
        if is_plumed_file(filename):
            df_tmp = plumed_to_pandas(filename)
            df_tmp["walker"] = [i for _ in range(len(df_tmp))]
            df_tmp = df_tmp.iloc[start:stop:stride, :]
            df_list.append(df_tmp)

        # else use read_csv with optional kwargs
        else:
            df_tmp = pd.read_csv(filename, **kwargs)
            df_tmp["walker"] = [i for _ in range(len(df_tmp))]
            df_tmp = df_tmp.iloc[start:stop:stride, :]
            df_list.append(df_tmp)

        # delete temporary data if necessary
        if download:
            if delete_download:
                os.remove(filename)
            else:
                print(f"downloaded file ({url}) saved as ({filename}).")

        # concatenate
        df = pd.concat(df_list)
        df.reset_index(drop=True, inplace=True)

    return df


##########################################################################
## FESSA COLOR PALETTE
##########################################################################
#  https://github.com/luigibonati/fessa-color-palette/blob/master/fessa.py

from matplotlib.colors import LinearSegmentedColormap, ColorConverter
import matplotlib as mpl
import matplotlib.pyplot as plt

# Fessa colormap
paletteFessa = [
    "#1F3B73",  # dark-blue
    "#2F9294",  # green-blue
    "#50B28D",  # green
    "#A7D655",  # pisello
    "#FFE03E",  # yellow
    "#FFA955",  # orange
    "#D6573B",  # red
]

cm_fessa = LinearSegmentedColormap.from_list("fessa", paletteFessa)
mpl.colormaps.register(cmap=cm_fessa)
mpl.colormaps.register(cmap=cm_fessa.reversed())

for i in range(len(paletteFessa)):
    ColorConverter.colors[f"fessa{i}"] = paletteFessa[i]

### To set it as default
# import fessa
# plt.set_cmap('fessa')
### or the reversed one
# plt.set_cmap('fessa_r')
### For contour plots
# plt.contourf(X, Y, Z, cmap='fessa')
### For standard plots
# plt.plot(x, y, color='fessa0')

##########################################################################
## CHEMISCOPE
##########################################################################


def create_chemiscope_input(
    trajectory, filename=None, colvar=None, cvs=["*"], verbose=False
):
    """
    Create a chemiscope input file from a trajectory and optional collective variables (colvar) file.

    Parameters
    ----------
    trajectory : list of ase.Atoms or str
        Trajectory of atoms objects or path to an xyz file
    filename : str, optional
        Output filename. If None, it will be saved with the same name of the trajectory with _chemiscope.json.gz appended.
    colvar : str or pandas.DataFrame, optional
        Path of the COLVAR file or a pandas dataframe
    cvs : list of str, optional
        List of collective variable names to be saved into the chemiscope file. If a string contains '*', it will be used as a filter for the property names in the colvar file (e.g. 'cv.*' will extract all properties with 'cv.' prefix). Default is ['*']
    verbose: bool, optional
        Print information
    Returns
    -------
    filename : str
        Path of the chemiscope input file
    """

    if verbose:
        print("[INFO] Creating Chemiscope input file...")
    try:
        import chemiscope
    except ImportError:
        raise ImportError(
            "Chemiscope is not installed. Please install it with pip install chemiscope"
        )

    # check if trajectory is a list of atoms or a filename
    if isinstance(trajectory, list) & isinstance(trajectory[0], Atoms):
        traj = trajectory
    elif isinstance(trajectory, str):
        if verbose:
            print("[INFO] Reading file:", trajectory)
        traj = read(trajectory, index=":")
    atoms = traj[0]

    # load colvar file into traj if requested
    if colvar is not None:
        if isinstance(colvar, pd.DataFrame):
            pass
        else:
            try:
                colvar = load_dataframe(colvar)

                # if time in colvar use colvar.time and atoms.info['frame'] to ensure consistency
                dt, consistent = None, True
                atoms_old = traj[0]
                if "time" in colvar.columns and "frame" in atoms_old.info:
                    # check if time is consistent between colvar and traj
                    if verbose:
                        print(
                            f"[INFO] Checking time consistency between COLVAR and trajectory..."
                        )
                    for i, atoms in enumerate(traj[1:]):
                        if dt is None:
                            frames = atoms.info["frame"] - atoms_old.info["frame"]
                            time_interval = (
                                colvar["time"].loc[atoms.info["frame"]]
                                - colvar["time"].loc[atoms_old.info["frame"]]
                            )
                            dt = time_interval / frames
                        else:  # check consistency
                            frames = atoms.info["frame"] - atoms_old.info["frame"]
                            time_interval = (
                                colvar["time"].loc[atoms.info["frame"]]
                                - colvar["time"].loc[atoms_old.info["frame"]]
                            )
                            dt_new = time_interval / frames
                            if np.abs(dt - dt_new) > 1e-8:
                                consistent = False
                        atoms_old = atoms
                    if consistent:
                        if verbose:
                            print(
                                f"[INFO] Time consistency between COLVAR and trajectory verified."
                            )
                        for i, atoms in enumerate(traj):
                            for col in colvar.columns:
                                atoms.info["colvar." + col] = colvar[col].loc[
                                    atoms.info["frame"]
                                ]
                    else:
                        print(
                            f"[WARNING]: time inconsistency between COLVAR and trajectory detected. Not saving COLVAR information."
                        )
                        for i, atoms in enumerate(traj):
                            for col in colvar.columns:
                                atoms.info["colvar." + col] = colvar[col].iloc[i]
                elif len(colvar) == len(traj):
                    print(
                        "[WARNING]: Consistency between traj and COLVAR cannot be assessed. Saving COLVAR assuming that the order of frames in COLVAR and trajectory are the same."
                    )
                    for i, atoms in enumerate(traj):
                        for col in colvar.columns:
                            atoms.info["colvar." + col] = colvar[col].iloc[i]
                else:
                    print(
                        "[WARNING]: Consistency between traj and COLVAR cannot be assessed and lengths do not match. Not saving COLVAR information."
                    )
            except Exception as e:
                print(
                    f"[WARNING]: colvar file: {colvar} not read, it should be a string filename or a pandas dataframe. Exception: {e}."
                )

    # Get CV names
    prop_names, prop_names_float = [], []
    for c in cvs:
        if "*" in c:
            prop_names.extend([p for p in atoms.info.keys() if c.replace("*", "") in p])
        else:
            prop_names.append(c)

    # Check if CV names can be converted to float
    for p in prop_names:
        if p == "target_atoms":
            continue
        try:
            float(atoms.info[p])
            prop_names_float.append(p)
        except TypeError:
            # if p != "target_atoms":
            print(f'skipping "{p}" as it cannot be converted to float.')

    if verbose:
        print("[INFO] CV names:", prop_names_float)

    # Extract properties
    properties = chemiscope.extract_properties(traj, only=prop_names_float)

    # Define shape and colors
    shapes_selection = []
    for atoms in traj:
        target_atoms = atoms.info.get("target_atoms", None)
        for i, atom in enumerate(atoms):
            if target_atoms is not None:
                if not isinstance(target_atoms, np.ndarray):
                    target_atoms = np.asarray([target_atoms])
                shapes_selection.append(
                    {
                        "radius": CR[CS.index(atom.symbol)],
                        "color": None if i in target_atoms else "#d4d4d4",
                    }
                )
            else:
                shapes_selection.append(
                    {"radius": CR[CS.index(atom.symbol)], "color": None}
                )
    if target_atoms is not None and verbose:
        print('[INFO] "target_atoms" found in atoms.')

    # Write input
    if filename is None:
        if isinstance(trajectory, str):
            filename = os.path.splitext(trajectory)[0] + "_chemiscope.json.gz"
        else:
            filename = "chemiscope.json.gz"

    chemiscope.write_input(
        filename,
        frames=traj,
        properties=properties,
        meta=dict(name="DEAL selection"),
        shapes={
            "selection": {"kind": "sphere", "parameters": {"atom": shapes_selection}},
        },
        settings={
            "structure": [
                {
                    "atoms": False,
                    "bonds": False,
                    "shape": "selection",
                    "axes": "off",
                    "keepOrientation": False,
                    "playbackDelay": 700,
                }
            ]
        },
    )

    if verbose:
        print("[OUTPUT] Chemiscope input saved in:", filename)

    return filename
