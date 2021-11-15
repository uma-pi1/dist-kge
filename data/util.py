import os
import numpy as np
import pandas as pd
from typing import List
from shutil import copyfile

S, P, O = [0, 1, 2]


def read_data(folder, train=True, valid=False, test=False, entity_ids=False, relation_ids=False):
    returns = []
    if train:
        print("read train.del")
        if os.path.exists(os.path.join(folder, "train.del.npy")):
            print("read pickle file")
            train = np.load(os.path.join(folder, "train.del.npy"))
            returns.append(train)
        else:
            train = pd.read_csv(
                        os.path.join(folder, "train.del"), sep="\t", dtype=int, header=None,
                        usecols=range(0, 3)
                    ).to_numpy()
            print("write pickle file")
            # np.save(os.path.join(folder, "train.del"), train)
            returns.append(train)
    if valid:
        print("read valid.del")
        valid = pd.read_csv(
            os.path.join(folder, "valid.del"), sep="\t", dtype=int, header=None,
            usecols=range(0, 3)
        ).to_numpy()
        returns.append(valid)
    if test:
        print("read test.del")
        test = pd.read_csv(
            os.path.join(folder, "test.del"), sep="\t", dtype=int, header=None,
            usecols=range(0, 3)
        ).to_numpy()
        returns.append(test)
    if entity_ids:
        print("read entity_ids.del")
        entity_ids = pd.read_csv(
            os.path.join(folder, "entity_ids.del"), sep="\t", dtype=str, header=None,
        ).to_numpy()
        returns.append(entity_ids)
    if relation_ids:
        print("read relation_ids.del")
        relation_ids = pd.read_csv(
            os.path.join(folder, "relation_ids.del"), sep="\t", dtype=str, header=None,
        ).to_numpy()
        returns.append(relation_ids)
    return tuple(returns)


def write_to_file(output_folder, train=None, valid=None, test=None, entity_ids=None, relation_ids=None):
    print("write to file")
    if not os.path.isdir(output_folder):
        try:
            os.mkdir(output_folder)
        except OSError:
            print("Creation of the directory {} failed".format(output_folder))
        else:
            print("Successfully created the directory {}".format(output_folder))
    if train is not None:
        np.savetxt(
            os.path.join(output_folder, "train.del"),
            train,
            delimiter="\t",
            fmt="%d",
        )
    if valid is not None:
        np.savetxt(
            os.path.join(output_folder, "valid.del"),
            valid,
            delimiter="\t",
            fmt="%d",
        )
    if test is not None:
        np.savetxt(
            os.path.join(output_folder, "test.del"),
            test,
            delimiter="\t",
            fmt="%d",
        )
    if entity_ids is not None:
        np.savetxt(
            os.path.join(output_folder, "entity_ids.del"),
            entity_ids,
            delimiter="\t",
            fmt="%s",
        )
    if relation_ids is not None:
        np.savetxt(
            os.path.join(output_folder, "relation_ids.del"),
            relation_ids,
            delimiter="\t",
            fmt="%s",
        )
