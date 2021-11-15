import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import KgeModel, KgeEmbedder
from kge.util import load_checkpoint
from kge.distributed.misc import get_min_rank
from typing import Optional
from copy import deepcopy


class DistributedModel(KgeModel):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
        create_embedders=True,
        parameter_client=None,
        max_partition_entities=0,
    ):
        self._init_configuration(config, configuration_key)
        self.base_model_config_key = self.configuration_key + ".base_model"
        self.parameter_client = parameter_client
        self.max_partition_entities = max_partition_entities
        base_config = deepcopy(config)
        base_config.set("model", config.get("distributed_model.base_model.type"))
        base_model = KgeModel.create(
            config=base_config,
            dataset=dataset,
            configuration_key=self.base_model_config_key,
            init_for_load_only=init_for_load_only,
            create_embedders=False,
        )
        # Initialize this model
        super().__init__(
            config=config,
            dataset=dataset,
            create_embedders=False,
            scorer=base_model.get_scorer(),
            init_for_load_only=init_for_load_only,
        )
        self.base_model = base_model
        if create_embedders:
            self._create_embedders(init_for_load_only, parameter_client, max_partition_entities)

    def _create_embedders(self, init_for_load_only, parameter_client=None, max_partition_entities=0):
        # if self.get_option("create_complete"):
        #    embedding_layer_size = dataset.num_entities()
        if self.config.get(
                "job.distributed.entity_sync_level") == "partition" and max_partition_entities != 0:
            embedding_layer_size = max_partition_entities
        else:
            embedding_layer_size = self._calc_embedding_layer_size(self.config, self.base_model.dataset)
        self.config.log(f"creating entity_embedder with {embedding_layer_size} keys")
        self._entity_embedder = KgeEmbedder.create(
            config=self.base_model.config,
            dataset=self.base_model.dataset,
            configuration_key=self.base_model_config_key + ".entity_embedder",
            vocab_size=embedding_layer_size,
            init_for_load_only=init_for_load_only,
            parameter_client=parameter_client,
            lapse_offset=0,
            complete_vocab_size=self.base_model.dataset.num_entities()
        )

        #: Embedder used for relations
        num_relations = self.base_model.dataset.num_relations()
        self._relation_embedder = KgeEmbedder.create(
            self.base_model.config,
            self.base_model.dataset,
            self.base_model_config_key + ".relation_embedder",
            num_relations,
            init_for_load_only=init_for_load_only,
            parameter_client=parameter_client,
            lapse_offset=self.base_model.dataset.num_entities(),
            complete_vocab_size=self.base_model.dataset.num_relations(),
            )

        if not init_for_load_only and parameter_client.rank == get_min_rank(self.config):
            # load pretrained embeddings
            pretrained_entities_filename = ""
            pretrained_relations_filename = ""
            if self.base_model.has_option("entity_embedder.pretrain.model_filename"):
                pretrained_entities_filename = self.base_model.get_option(
                    "entity_embedder.pretrain.model_filename"
                )
            if self.base_model.has_option("relation_embedder.pretrain.model_filename"):
                pretrained_relations_filename = self.base_model.get_option(
                    "relation_embedder.pretrain.model_filename"
                )

            def load_pretrained_model(
                    pretrained_filename: str,
            ) -> Optional[KgeModel]:
                if pretrained_filename != "":
                    self.config.log(
                        f"Initializing with embeddings stored in "
                        f"{pretrained_filename}"
                    )
                    checkpoint = load_checkpoint(pretrained_filename)
                    return KgeModel.create_from(checkpoint,
                                                parameter_client=parameter_client)
                return None

            pretrained_entities_model = load_pretrained_model(
                pretrained_entities_filename
            )
            if pretrained_entities_filename == pretrained_relations_filename:
                pretrained_relations_model = pretrained_entities_model
            else:
                pretrained_relations_model = load_pretrained_model(
                    pretrained_relations_filename
                )
            if pretrained_entities_model is not None:
                if (
                        pretrained_entities_model.get_s_embedder()
                        != pretrained_entities_model.get_o_embedder()
                ):
                    raise ValueError(
                        "Can only initialize with pre-trained models having "
                        "identical subject and object embeddings."
                    )
                self._entity_embedder.init_pretrained(
                    pretrained_entities_model.get_s_embedder()
                )
            if pretrained_relations_model is not None:
                self._relation_embedder.init_pretrained(
                    pretrained_relations_model.get_p_embedder()
                )
