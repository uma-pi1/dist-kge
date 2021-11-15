import argparse
import numpy as np
import numba
import os
import math
import torch
from util import *
from typing import Dict, Tuple
from pathlib import Path


def shuffle_with_inverse(data: np.array):
    order = np.arange(len(data))
    np.random.shuffle(order)
    inverse = np.argsort(order)
    return data[order], inverse


def _construct_partitions_slow(partition_assignment, num_partitions):
    partition_indexes = np.unique(partition_assignment, axis=0)
    partitions_data = [
        torch.from_numpy(
            np.where(np.all(partition_assignment == i, axis=1))[0]
        ).contiguous()
        for i in partition_indexes
    ]
    partition_indexes = [(i[0], i[1]) for i in partition_indexes]
    partitions = dict(zip(partition_indexes, partitions_data))
    return partitions


def _construct_partitions(partition_assignment, num_partitions):
    partition_indexes, partition_data = _numba_construct_partitions(np.ascontiguousarray(partition_assignment), num_partitions)
    partition_indexes = [(i, j) for i in range(num_partitions) for j in range(num_partitions)]
    partition_data = [torch.from_numpy(data).long().contiguous() for data in partition_data]
    partitions = dict(zip(partition_indexes, partition_data))
    return partitions


@numba.njit
def _numba_construct_partitions(partition_assignment, num_partitions):
    partition_indexes = [
        (i, j) for i in range(num_partitions) for j in range(num_partitions)
    ]
    partition_id_lookup: Dict[Tuple[int, int], int] = dict()
    partition_lengths: Dict[int, int] = dict()
    partition_data = []
    for i in range(len(partition_indexes)):
        partition = partition_indexes[i]
        partition_id_lookup[partition] = i
        partition_lengths[i] = 0
        partition_data.append(
            np.empty(
                int(len(partition_assignment)/num_partitions),
                dtype=np.int64
            )
        )

    # iterate over the partition assignments and assign each triple-id to its
    #  corresponding partition
    for i in range(len(partition_assignment)):
        pa = partition_assignment[i]
        pa_tuple = (pa[0], pa[1])
        partition_id = partition_id_lookup[pa_tuple]
        current_partition_size = partition_lengths[partition_id]
        partition_data[partition_id][current_partition_size] = i
        partition_lengths[partition_id] += 1

    # now get correct sizes of partitions
    for i in range(len(partition_data)):
        partition_data[i] = partition_data[i][:partition_lengths[i]]
    return partition_indexes, partition_data


def _get_entities_in_bucket(entities_to_partition, partitions):
    entities_in_bucket = dict()
    for partition in partitions:
        entities_in_bucket[partition] = torch.from_numpy(
            np.where(
                np.ma.mask_or(
                    (entities_to_partition == partition[0]),
                    (entities_to_partition == partition[1]),
                )
            )[0]
        )
    return entities_in_bucket


def random_map_entities(data: np.array, entity_ids: np.array):
    mapper = np.arange(len(entity_ids)).astype(np.long)
    np.random.shuffle(mapper)
    data[:, 0] = mapper[data[:, 0]]
    data[:, 2] = mapper[data[:, 2]]
    mapped_entity_ids = mapper[entity_ids[:, 0].astype(np.long)]
    return data, mapped_entity_ids


def get_partition_old(entity_id, dataset_size, num_partitions):
    partition = math.floor(entity_id * 1.0 / dataset_size * 1.0 * num_partitions)
    return partition

@numba.guvectorize([(numba.int64[:], numba.int64, numba.int64, numba.int64[:])], '(n),(),()->(n)')
def get_partition(entity_ids, dataset_size, num_partitions, res):
    # result = np.empty(len(entity_ids), dtype=np.long)
    # num_partitions = 8
    for i in range(len(entity_ids)):
        res[i] = int(
            entity_ids[i] * 1.0 / dataset_size * 1.0 * num_partitions
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of dataset to partition")
    parser.add_argument("-n", "--num_entity_partitions", type=int, help="number of entity partitions. Will create n^2 partitions.")
    args = parser.parse_args()
    dataset_folder = args.dataset
    write_to_file = True
    num_partitions = args.num_entity_partitions
    train_data, entity_ids = read_data(dataset_folder, entity_ids=True)
    # most often occurring entities are usually the ones with low ids
    #  to make partitions of similar sizes we are randomly mapping the ids before
    #  assigning the partitions
    mapped_data, mapped_entity_ids = random_map_entities(train_data, entity_ids)
    #v_get_partition = np.vectorize(
    #    get_partition, excluded=["dataset_size", "num_partitions"]
    #)
    print("partition subject")
    s_block = get_partition(
        mapped_data[:, 0], len(entity_ids), num_partitions
    )
    print("partition object")
    o_block = get_partition(
        mapped_data[:, 2], len(entity_ids), num_partitions
    )
    print("map entity ids to partition")
    entity_to_partition = get_partition(mapped_entity_ids, len(entity_ids), num_partitions)
    triple_partition_assignment = np.ascontiguousarray(np.stack([s_block, o_block], axis=1))
    partition_folder = os.path.join(dataset_folder, "partitions", "stratification")
    output_folder = os.path.join(partition_folder, f"num_{num_partitions}")
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    if write_to_file:
        print("write to file")
        np.savetxt(
            os.path.join(output_folder, "train_assign_partitions.del"),
            triple_partition_assignment,
            delimiter="\t",
            fmt="%s",
        )
        np.savetxt(
            os.path.join(output_folder, "entity_to_partitions.del"),
            entity_to_partition,
            delimiter="\t",
            fmt="%s",
        )

    print("construct partitions")
    partitions = _construct_partitions(triple_partition_assignment, num_partitions)
    entities_in_strata = _get_entities_in_bucket(entity_to_partition, partitions)

    print("\nactually used entities per block")
    overall_percentage_used = 0
    for strata, strata_data in partitions.items():
        entities = entities_in_strata[strata]
        used_entities = np.unique(train_data[strata_data][:, [0,2]])
        percentage_used = round(len(used_entities)/len(entities), 2)
        percentage_of_data = len(strata_data)/len(train_data)
        overall_percentage_used += percentage_of_data * percentage_used
        #print(len(entities), len(used_entities), percentage_used)
    print("overall percentage used", overall_percentage_used)

    print("\nactually used entities per combined block")
    overall_percentage_used_combined = 0
    for n1 in range(num_partitions):
        for n2 in range(n1, num_partitions):
            strata1 = (n1, n2)
            entities = entities_in_strata[strata1]
            if n1 == n2:
                if n1 % 2 == 0:
                    continue
                strata2 = (n1-1, n1-1)
                entities = np.concatenate((entities, entities_in_strata[strata2]))
            else:
                strata2 = (n2, n1)
            strata1_indices = partitions[strata1]
            strata2_indices = partitions[strata2]
            mirror_indices = np.concatenate((strata1_indices, strata2_indices))
            used_entities = np.unique(train_data[mirror_indices][:, [0,2]])
            percentage_used = round(len(used_entities)/len(entities), 2)
            percentage_of_data = len(mirror_indices)/len(train_data)
            # print("strata1", strata1, "strata2", strata2)
            overall_percentage_used_combined += percentage_of_data * percentage_used
            # print(len(entities), len(used_entities), percentage_used)
    print("overall percentage used combined", overall_percentage_used_combined)


