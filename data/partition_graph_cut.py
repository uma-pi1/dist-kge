import os
import argparse
import dgl
import torch
import numpy as np
import scipy as sp
from pathlib import Path
from util import read_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="name of dataset to partition")
    parser.add_argument("-n", "--num_partitions", type=int, help="number of partitions")
    args = parser.parse_args()
    dataset_folder = args.dataset
    num_parts = args.num_partitions
    write=True
    data, entities, relations = read_data(dataset_folder, train=True, entity_ids=True, relation_ids=True)


    src = data[:, 0]
    dst = data[:, 2]
    num_entities = len(entities)
    coo = sp.sparse.coo_matrix((np.ones(data.shape[0]), (src, dst)),
                               shape=[num_entities, num_entities])

    triple_partition_assignment = np.full((len(data)), -1, dtype=np.int)
    entity_partition_assignment = np.full((num_entities), -1, dtype=np.int)
    num_inner_edges_dict = {}
    inner_nodes_dict = {}
    print("construct graph...")
    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
    g.edata['tid'] = torch.from_numpy(data[:, 1])
    print('partition graph...')
    part_dict = dgl.transform.metis_partition(g, num_parts, 1)

    node_part_mapper = np.zeros(num_entities)-1

    tot_num_inner_edges = 0
    print("partition\t|\tentities needed\t|\tentities located in partition\t|\tnum triples\t|\tinner triples\t|\tnum outside partition acesses\t|\tpercent outside partition acesses")
    print("--------:\t|\t----:\t|\t--------:\t|\t------:\t|\t----------:\t|\t---------:\t|\t---------:")
    for part_id in part_dict:
        part = part_dict[part_id]
        #print(part.has_nodes(entities))
        src, dst = part.all_edges(form='uv', order='eid')
        #print(src)
        triple_partition_assignment[part.edata["_ID"].numpy()] = part_id
        #entity_partition_assignment[part.ndata["_ID"].numpy()] = part_id


        num_inner_nodes = len(np.nonzero(part.ndata['inner_node'].numpy())[0])
        num_inner_edges = len(np.nonzero(part.edata['inner_edge'].numpy())[0])
        num_inner_edges_dict[part_id] = num_inner_edges
        outside_partition_accesses = part.number_of_edges() - num_inner_edges
        print("{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t|\t{}\t".format(
            part_id, part.number_of_nodes(), num_inner_nodes,
            part.number_of_edges(), num_inner_edges,
            outside_partition_accesses,
            outside_partition_accesses/part.number_of_edges()
        ))
        tot_num_inner_edges += num_inner_edges

        part.copy_from_parent()
        parent_inner_nodes = part.parent_nid[part.ndata["inner_node"].numpy().astype(bool)]
        inner_nodes_dict[part_id] = parent_inner_nodes
        mapper = node_part_mapper[parent_inner_nodes]
        mapper[mapper==-1] = part_id
        #print(mapper)
        #node_part_mapper[part.parent_nid] = mapper
        entity_partition_assignment[parent_inner_nodes] = mapper
        #print(node_part_mapper)

    # write to file

    if write:
        partition_folder = os.path.join(dataset_folder, "partitions","graph-cut")
        output_folder = os.path.join(partition_folder, f"num_{num_parts}")
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        print("write to file")
        np.savetxt(
            os.path.join(output_folder, "train_assign_partitions.del"),
            triple_partition_assignment,
            delimiter="\t",
            fmt="%s",
        )
        np.savetxt(
            os.path.join(output_folder, "entity_to_partitions.del"),
            entity_partition_assignment,
            delimiter="\t",
            fmt="%s",
        )
    print("some counting")
    unique_pairs, counts = np.unique(
        triple_partition_assignment, return_counts=True, axis=0
    )
    print(unique_pairs)
    print(counts)

    print(triple_partition_assignment)
