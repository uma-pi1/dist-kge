import os
import argparse
from tqdm import tqdm
import numpy as np
from util import *
from pathlib import Path

S, P, O = [0, 1, 2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of dataset to partition")
    parser.add_argument("-n", "--num_partitions", type=int, help="number of partitions")
    args = parser.parse_args()
    dataset_folder = args.dataset
    train_data, relation_ids = read_data(dataset_folder, relation_ids=True)
    num_partitions = args.num_partitions
    partition_mapper = np.zeros(len(train_data), dtype=np.int) - 1
    relation_partition_mapper = np.full(len(relation_ids), -1, dtype=np.int)
    partition_sizes = np.zeros(num_partitions, dtype=np.int)
    unique_relations, relation_counts = np.unique(train_data[:, P], return_counts=True)
    relation_descending_sorter = np.argsort(relation_counts*-1)
    relation_counts_sorted = relation_counts[relation_descending_sorter]
    print("assigning relations to partitions")
    for i, relation in enumerate(tqdm(unique_relations[relation_descending_sorter])):
        current_smallest_partition = np.argmin(partition_sizes)
        partition_mapper[train_data[:, P] == relation] = current_smallest_partition
        partition_sizes[current_smallest_partition] += relation_counts_sorted[i]
        relation_partition_mapper[relation] = current_smallest_partition

    partition_folder = os.path.join(dataset_folder, "partitions", "relation")
    output_folder = os.path.join(partition_folder, f"num_{num_partitions}")
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    print("write to file")
    np.savetxt(
        os.path.join(output_folder, "train_assign_partitions.del"),
        partition_mapper,
        delimiter="\t",
        fmt="%s",
    )
    np.savetxt(
        os.path.join(output_folder, "relation_to_partitions.del"),
        relation_partition_mapper,
        delimiter="\t",
        fmt="%s",
    )
    print("some counting")
    unique_pairs, counts = np.unique(
        partition_mapper, return_counts=True, axis=0
    )
    print(unique_pairs)
    print(counts)

    print(partition_mapper)
