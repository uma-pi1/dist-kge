# Dist-KGE: A knowledge graph embedding library for multi-GPU and multi-machine training


This is the code and configuration accompanying the paper ["Parallel Training of Knowledge Graph Embedding Models: A Comparison of Techniques"](https://vldb.org/pvldb/vol15/p633-kochsiek.pdf).
The code extends the knowledge graph embedding library [LibKGE](https://github.com/uma-pi1/kge).
For documentation on LibKGE refer to LibKGE repository.
We provide the hyper-parameter settings for the experiments in their corresponding configuration files.


## Table of contents

1. [Quick start](#quick-start)
2. [Dataset preparation for parallel training](#dataset-preparation-for-parallel-training)
3. [Single Machine Multi-GPU Training](#single-machine-multi-gpu-training)
4. [Multi-GPU Multi-Machine Training](#multi-gpu-multi-machine-training)
5. [Folder structure of experiment results](#folder-structure-of-experiment-results)
6. [Results and Configurations](#results-and-configurations)
7. [How to cite](#how-to-cite)

## Quick start

```sh
# retrieve and install project in development mode
git clone https://github.com/uma-pi1/dist-kge.git
cd dist-kge
pip install -e .

# download and preprocess datasets
cd data
sh download_all.sh
cd ..

# train an example model on toy dataset (you can omit '--job.device cpu' when you have a gpu)
kge start examples/toy-complex-train.yaml --job.device cpu

```
This example will train on a toy dataset in a sequential setup on CPU.

For further documentation on possible configuration can be found in the file [config-default.yaml](kge/config-default.yaml).

#### Supported Models
For a list of models for sequential training using a single GPU we refer to [LibKGE](https://github.com/uma-pi1/kge).

Models supporte for Multi-GPU and Multi-Machine are:
- ComplEx
- RotatE
- TransE


## Dataset preparation for parallel training
**NOTE: Freebase already comes with multiple partition settings to save preprocessing time**

To partition the data run the following commands (you only need to do this once)

**Random Partitioning**

For random partitioning no further preparation is needed.

**Relation Partitioning**
```sh
cd data
python partition_relation.py <dataset-name> -n <num-partitions>
cd ..
```

**Stratification**
```sh
cd data 
python partition_stratification.py <dataset-name> -n <num-partitions>
cd ..
```

**Graph-Cut**

````sh
cd data
python partition_graph_cut.py <dataset-name> -n <num-partitions>
cd ..
````


## Single Machine Multi-GPU Training
Run following example to train on two GPUs with random partitioning (two worker per GPU):
````
python -m kge start examples/fb15k-complex-parallel.yaml
````
The most important configuration options for multi-gpu training are:
````yaml
import:
  - complex
  - distributed_model
model: distributed_model
distributed_model:
  base_model: complex
job:
  distributed:
    num_partitions: 4
    num_workers: 4
    partition_type: random
    master_port: 8888  # change in case this port is used on your machine
  device_pool:
    - cuda:0
    - cuda:1
train:
  type: distributed_negative_sampling
  optimizer:
    default:
      type: dist_adagrad
````

## Multi-GPU Multi-Machine Training
### Parameter Server
For multi-machine training we use the parameter server [Lapse](https://github.com/alexrenz/lapse-ps).
To install Lapse and the corresponding python bindings run the following commands:
````sh
git clone https://github.com/alexrenz/lapse-ps.git lapse-ps
cd lapse-ps
git checkout 72c7197c6d1795a8de184de84ffbe1b08151756d
make ps KEY_TYPE=int64_t CXX11_ABI=$(python bindings/lookup_torch_abi.py) DEPS_PATH=$(pwd)/deps_bindings
cd bindings 
python setup.py install --user
````
For further documentation on the python bindings refer to [Lapse-Binding documentation](https://github.com/alexrenz/lapse-ps/tree/main/bindings).

In case you can not use Lapse, we provide a very inefficient parameter server (for debugging). To use this debugging PS use the option `--job.distributed.parameter_server torch`

### Interface
As we use the gloo backend to communicate between master and worker nodes you need to specify the interface connecting your machines and specify it as `--job.distributed.gloo_socket_ifname`.
You can find out the names of your interfaces with the command
````sh
ip address
````

### Example
Run the following example to train on two machines with one GPU each (1@2) with random partitioning:

Command for machine 1
````sh
python -m kge start examples/fb15k-complex-distributed.yaml --job.distributed.machine_id 0 --job.distributed.master_ip <ip_of_machine_0>
````

Command for machine 2
````sh
python -m kge start examples/fb15k-complex-distributed.yaml --job.distributed.machine_id 1 --job.distributed.master_ip <ip_of_machine_0>
````


Important options for distributed training in addition to the options specified in the single-machine setting are:
````yaml
job:
  distributed:
    master_ip: '<ip_of_machine_0>'  # ip address of the machine with machine_id 0
    num_machines: 2
    num_workers: 4  # total number of workers over all machines
    gloo_socket_ifname: bond0  # name of the interface to use. Use command 'ip address' to find names
    parameter_server: lapse
````

#### Different number of workers per machine
If you have two machines with a varying number of GPUs you might want to set a varying number of workers per machine. 
You can do so with the following commands

Command for machine 1
````sh
python -m kge start examples/fb15k-complex-distributed.yaml --job.distributed.machine_id 0 --job.distributed.master_ip <ip_of_machine_0> --job.distributed_num_workers_machine 3
````

Command for machine 2
````sh
python -m kge start examples/fb15k-complex-distributed.yaml --job.distributed.machine_id 1 --job.distributed.master_ip <ip_of_machine_0> --job.distributed.num_workers_machine 1 --job.distributed.already_init_workers 3
````

Note that you need to specify for each machine how many workers are initialized on the previous machines with `job.distributed.already_init_workers` if the number of workers varies per machine.
You can also create one configuration file per machine, with corresponding settings for `num_workers_machine` and `already_init_workers`.

## Folder structure of experiment results
- by default, each experiment will create a new folder in `local/experiments/<timestamp>-<config-name>`
- this folder can be changed with command line argument `--folder path/to/folder`
- for multi-machine training a folder is created for each machine. Therefore, specify a separate folder name for each machine if you work on a shared filesystem.
- each worker will have its own subfolder logging partition-processing times
- the complete epoch time over all partitions is logged in the main `kge.log` file
- hardware information is logged into `hardware_monitor.log` and `gpu_monitor.log`
- evaluation is performed on machine-0 by worker-0. Therefore, evaluation results are logged into folder `<experiment-folder-machine-0>/worker-0/` in the files `kge.log` and `trace.yaml`


## Results and Configurations
- all ranking metrics are filtered with train, valid and test if not mentioned otherwise
- configuration files for the experiments can be found [here](examples/experiments)

### Partitioning techniques (best-performing variant)
- best performing variant in terms of time to 0.95 MRR reached in the sequential setting

#### FB15k
**ComplEx**

Setup   |   Partitioning Technique  |   Epoch Time  |   Time to 0.95 MRR    |   MRR |   MRR unfiltered  |   Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   -----
. |  Sequential (GPU memory)    |   5.9s    |   3.9min  |   0.778   |  0.245 | 0.729  |   0.862  |   0.932  |   [config](examples/experiments/fb15k/dim128/complex/complex-fb15k-sequential.yaml)
. |  Sequential (main memory)    |   7.7s    |   5.1min  |   0.778   | 0.245  |  0.729 |   0.862    |   0.932   |   [config](examples/experiments/fb15k/dim128/complex/complex-fb15k-sequential-mm.yaml)
2@1 |  Random (R)    |   2.6s    |   2.0min  |   0.775   |  0.243 | 0.726  |   0.859    |   0.931   |   [config](examples/experiments/fb15k/dim128/complex/complex-fb15k-parallel-random-R-2@1.yaml)
1@2 |  Random (R)    |   2.9s    |   2.2min  |   0.775   |  0.243 | 0.726  |   0.859    |   0.931   |   [config](examples/experiments/fb15k/dim128/complex/complex-fb15k-distributed-random-R-1@2.yaml)
4@2 |  Random (R)    |   1.3s    |   1.3min  |   0.766   |  0.241 |  0.712 |   0.858    |   0.929   |   [config](examples/experiments/fb15k/dim128/complex/complex-fb15k-distributed-random-R-4@2.yaml)

**RotatE**

Setup   |   Partitioning Technique  |   Epoch Time  |   Time to 0.95 MRR    |   MRR |   MRR unfiltered  |   Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   -----
. |  Sequential (GPU memory)    |   9.5s    |   11.9min  |   0.705   | 0.232  |  0.630 |   0.834    |   0.928   |   [config](examples/experiments/fb15k/dim128/rotate/rotate-fb15k-sequential.yaml)
. |  Sequential (main memory)    |   11.4s    |   14.3min  |   0.705   |  0.232 |  0.630 |   0.834    |   0.928 |   [config](examples/experiments/fb15k/dim128/rotate/rotate-fb15k-sequential-mm.yaml)
2@1 |  Stratification (CARL)    |   4.6s    |   5.8min  |   0.725   | 0.239  |  0.664 |   0.835 |   0.926   |   [config](examples/experiments/fb15k/dim128/rotate/rotate-fb15k-parallel-stratification-CARL-2@1.yaml)
1@2 |  Stratification (CARL)    |   5.9s    |   7.4min  |   0.725   |  0.239 |  0.664 |   0.835 |   0.926   |   [config](examples/experiments/fb15k/dim128/rotate/rotate-fb15k-distributed-stratification-CARL-1@2.yaml)


#### Yago3-10
**ComplEx**

Setup   |   Partitioning Technique  |   Epoch Time  |   Time to 0.95 MRR    |   MRR |   MRR unfiltered  |   Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   -----
. |  Sequential (GPU memory)    |   24.3s    |   35.5min  |   0.542   | 0.111  |  0.468 |   0.675 |  0.791 |   [config](examples/experiments/yago3-10/dim128/complex/complex-yago3-10-sequential.yaml)
. |  Sequential (main memory)    |   42.6s    |   67.5min  |   0.542   |  0.111 | 0.468  |   0.675  |   0.791   |   [config](examples/experiments/yago3-10/dim128/complex/complex-yago3-10-sequential-mm.yaml)
2@1 |  Relation    |   19.0s    |   33.2min  |   0.538   | 0.107  | 0.465  |   0.669  | 0.787  |   [config](examples/experiments/yago3-10/dim128/complex/complex-yago3-10-parallel-relation-2@1.yaml)
1@2 |  Random (RL)    |   19.5s    |   35.8min  |   0.547   |  0.109 |  0.473 |   0.679  |  0.791 |   [config](examples/experiments/yago3-10/dim128/complex/complex-yago3-10-distributed-random-RL-1@2.yaml)
4@2 |  Random (RL)    |   5.6s    |   n.r.  |   0.503   |  0.106 |  0.423 |   0.653    |   0.778   |   [config](examples/experiments/yago3-10/dim128/complex/complex-yago3-10-distributed-random-RL-4@2.yaml)


**RotatE**


Setup   |   Partitioning Technique  |   Epoch Time  |   Time to 0.95 MRR    |   MRR |   MRR unfiltered  |   Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   -----
. |  Sequential (GPU memory)    |   74.1s    |   259.3min  |   0.451   | 0.104  | 0.343  |   0.637 |  0.773 |   [config](examples/experiments/yago3-10/dim128/rotate/rotate-yago3-10-sequential.yaml)
. |  Sequential (main memory)    |   88.0s    |   307.8min  |   0.451   | 0.104  | 0.343  |   0.637 | 0.773  |   [config](examples/experiments/yago3-10/dim128/rotate/rotate-yago3-10-sequential-mm.yaml)
2@1 |  Stratification (CARL)    |   40.8s    |   166.6min  |   0.438   |  0.115 | 0.350  |   0.607 |  0.764 |  [config](examples/experiments/yago3-10/dim128/rotate/rotate-yago3-10-parallel-stratification-CARL-2@1.yaml)
1@2 |  Stratification (CARL)    |   43.3s    |   175.8min  |   0.438   |  0.115 |  0.350 |   0.607 | 0.764  |  [config](examples/experiments/yago3-10/dim128/rotate/rotate-yago3-10-distributed-stratification-CARL-1@2.yaml)


#### Wikidata5m

**ComplEx**

Setup   |   Partitioning Technique  |   Epoch Time  |   Time to 0.95 MRR    |   MRR |   MRR unfiltered  |   Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   -----
. |  Sequential (GPU memory)    |   438.4s    |   219.0min  |   0.297   |  0.255 | 0.246  |   0.385 | 0.516  |   [config](examples/experiments/wikidata5m/dim128/complex/complex-wikidata5m-sequential.yaml)
. |  Sequential (GPU memory)    |   774.3s    |   387.0min  |   0.297   | 0.255  |  0.246 |   0.386 | 0.516  |  [config](examples/experiments/wikidata5m/dim128/complex/complex-wikidata5m-sequential-mm.yaml)
2@1 |  Stratification (CARL)    |   232.8s    |   77.6min  |   0.308   | 0.264  | 0.255  |   0.398 | 0.513  |   [config](examples/experiments/wikidata5m/dim128/complex/complex-wikidata5m-parallel-stratification-CARL-2@1.yaml)
1@2 |  Stratification (CARL)    |   228.0s    |   76.0min  |   0.308   | 0.264  | 0.255  |   0.398 |  0.513 |   [config](examples/experiments/wikidata5m/dim128/complex/complex-wikidata5m-distributed-stratification-CARL-1@2.yaml)


**RotatE**

Setup   |   Partitioning Technique  |   Epoch Time  |   Time to 0.95 MRR    |   MRR |   MRR unfiltered  |   Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   -----
. |  Sequential (GPU memory)    |   798.4s    |   199.6min  |   0.258   |  0.225 | 0.202  |   0.348 | 0.453  |   [config](examples/experiments/wikidata5m/dim128/rotate/rotate-wikidata5m-sequential.yaml)
. |  Sequential (GPU memory)    |   985.7s    |   246.4min  |   0.258   | 0.225  | 0.202  |   0.348 | 0.453  |  [config](examples/experiments/wikidata5m/dim128/rotate/rotate-wikidata5m-sequential-mm.yaml)
2@1 |  Stratification (ARL)    |   466.7s    |   77.8min  |   0.264   | 0.230  |  0.213 |   0.344 |  0.410 |   [config](examples/experiments/wikidata5m/dim128/rotate/rotate-wikidata5m-parallel-stratification-ARL-2@1.yaml)
1@2 |  Stratification (ARL)    |   477.7s    |   79.6min  |   0.264   | 0.230  | 0.213  |   0.344 | 0.410  |   [config](examples/experiments/wikidata5m/dim128/rotate/rotate-wikidata5m-distributed-stratification-ARL-1@2.yaml)


#### Freebase

**ComplEx**

Setup   |   Partitioning Technique  |   Epoch Time  |   Data sent per epoch    |   sMRR |   sMRR unfiltered  |  MRR |   MRR unfiltered   |  Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ---:    |   ----:   |   -----
. | Sequential (main memory)    |   3929.0s |   -   |   0.811   | 0.776  |   0.364   | 0.311  | 0.298  |   0.487    |   0.618   |   [config](examples/experiments/freebase/dim128/complex/complex-freebase-sequential-mm.yaml)
. | Sequential (B) (main memory)    |   3925.2s |   -   |   0.815   | 0.782  |   0.426   | 0.345  |  0.370 |   0.528    |   0.642   |   [config](examples/experiments/freebase/dim128/complex/complex-freebase-sequential-B-mm.yaml)
2@2 | Random (RLB)    |   966.7s |   232.8GB   |   0.816   |  0.782 |   0.426   | 0.352 |    0.371 |   0.529    |  0.639 |   [config](examples/experiments/freebase/dim128/complex/random/complex-freebase-distributed-random-RLB-2@2.yaml)
2@2 | Relation (rLB)    |   823.8s |   205.9GB   |   0.801   |  0.770 |   0.397   |  0.326 |  0.339 |   0.507    |  0.631 |   [config](examples/experiments/freebase/dim128/complex/relation/complex-freebase-distributed-relation-rLB-2@2.yaml)
2@2 | Stratification (CARLB)    |   803.9s |   123.2GB   |   0.793   | 0.761  |   0.325   | 0.285  |  0.272 |   0.424    | 0.563  |   [config](examples/experiments/freebase/dim128/complex/stratification/complex-freebase-distributed-stratification-CARLB-2@2.yaml)
2@2 | Graph-cut (LB)    |   1170.6s |   42.5GB   |   0.789   |  0.761 |   0.407   | 0.335  | 0.351  |   0.512    | 0.624  |   [config](examples/experiments/freebase/dim128/complex/graph-cut/complex-freebase-distributed-graph-cut-LB-2@2.yaml)
4@2 | Random (RLB)    |   591.6s |   251.9GB   |   0.819   | 0.784  |   0.421   | 0.346  |  0.364 |   0.523    |  0.638 |   [config](examples/experiments/freebase/dim128/complex/random/complex-freebase-distributed-random-RLB-4@2.yaml)

**RotatE**

Setup   |   Partitioning Technique  |   Epoch Time  |   Data sent per epoch    |   sMRR |   sMRR unfiltered  |  MRR |   MRR unfiltered   |  Hits@1  |   Hits@10 |   Hits@100    | config
----    |   -----   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ----:   |   ---:    |   ----:   |   -----
.   | Sequential (main memory)    |   6495.7s |   -   |   0.774   | 0.748  |   0.566   | 0.426  | 0.529  |   0.627    |   0.677   | [config](examples/experiments/freebase/dim128/rotate/rotate-freebase-sequential-mm.yaml)
.   | Sequential (B) (main memory)    |   6184.0s |   -   |   0.812   |  0.774 |   0.560   |  0.422 |  0.521 |   0.623    |   0.674 | [config](examples/experiments/freebase/dim128/rotate/rotate-freebase-sequential-B-mm.yaml)
2@2 | Random (RLB)     |   1541.4s |   232.2GB   |   0.812   | 0.773  |   0.567   |  0.425 |  0.529 |   0.630|  0.678   | [config](examples/experiments/freebase/dim128/rotate/random/rotate-freebase-distributed-random-RLB-2@2.yaml)
2@2 | Relation (rLB)     |   1498.1s |   207.2GB   |   0.791   | 0.758  |   0.551   | 0.407  | 0.515  |   0.608 |   0.656   | [config](examples/experiments/freebase/dim128/rotate/relation/rotate-freebase-distributed-relation-rLB-2@2.yaml)
2@2 | Stratification (CARLB)     |   1416.1s |   123.3GB   |   0.734   | 0.720  |   0.529   | 0.395  | 0.491  |   0.592 |   0.641   | [config](examples/experiments/freebase/dim128/rotate/stratification/rotate-freebase-distributed-stratification-CARLB-2@2.yaml)
2@2 | Graph-cut (LB)     |   1867.9s |   44.5GB   |   0.775   | 0.749  |   0.560   | 0.410  | 0.526  |   0.616  |   0.654   | [config](examples/experiments/freebase/dim128/rotate/graph-cut/rotate-freebase-distributed-graph-cut-LB-2@2.yaml)




### Row-Adagrad
Row-wise optimizers treat each embedding as a single parameter
instead of each dimension of an embedding and therefore reduce storage and
communication overhead by about 50\%. 
We observed no negative influence on the resulting embedding quality for all
partitioning methods but graph-cut partitioning, where the drop was small but
noticeable. Overall, we found Row-Adagrad to be a suitable approach to reduce
storage and communication costs.
We report on ComplEx, 1@2.

#### Yago3-10

Partition Technique |   Data sent (Adagrad) |   MRR (Adagrad)   |   Data sent (Row-Adagrad) |   MRR (Row-Adagrad)
---------   |   -----:  |   -----:  |   ----:   |   ----:
Sequential  |   -   |   0.542   |   -   |   0.542
Random (R)  |   7.2GB   |   0.538   |   5.0GB   |   0.534
Relation    |   7.1GB   |   0.538   |   4.9GB   |   0.542
Stratification  |   0.4GB   |   0.531   |   0.2GB   |   0.539
Graph-cut   |   0.2GB   |   0.211   |   0.1GB   |   0.180

#### Wikidata5m

Partition Technique |   Data sent (Adagrad) |   MRR (Adagrad)   |   Data sent (Row-Adagrad) |   MRR (Row-Adagrad)
---------   |   -----:  |   -----:  |   ----:   |   ----:
Sequential  |   -   |   0.297   |   -   |   0.291
Random (R)  |   125.2GB |   0.296   |   65.7GB  |   0.298
Relation    |   123.8GB |   0.296   |   63.5GB  |   0.300
Stratification  |   15.0GB  |   0.308   |   7.4GB   |   0.306
Graph-cut   |   6.1GB   |   0.192   |   3.7GB   |   0.181


# How to cite
```
@article{kochsiek2021parallel,
  title={Parallel training of knowledge graph embedding models: a comparison of techniques},
  author={Kochsiek, Adrian and Gemulla, Rainer},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={3},
  pages={633--645},
  year={2021},
  publisher={VLDB Endowment}
}
```