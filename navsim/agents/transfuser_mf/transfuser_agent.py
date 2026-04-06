from typing import Any, List, Dict, Optional, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.transfuser_mf.transfuser_config import TransfuserConfig
from navsim.agents.transfuser_mf.transfuser_model import TransfuserModel
from navsim.agents.transfuser_mf.transfuser_callback import TransfuserCallback
from navsim.agents.transfuser_mf.transfuser_loss import transfuser_loss
from navsim.agents.transfuser_mf.transfuser_features import TransfuserFeatureBuilder, TransfuserTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class TransfuserAgent(AbstractAgent):
    """Agent interface for TransFuser baseline."""

    def __init__(
        self,
        config: TransfuserConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes TransFuser agent.
        :param config: global config of TransFuser agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr

        self._checkpoint_path = checkpoint_path
        self._transfuser_model = TransfuserModel(config)

    def clear_cache(self):
        """Inherited, see superclass."""
        self._transfuser_model.clear_cache()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[i for i in range(self._config.cam_seq_len)])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TransfuserTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [TransfuserFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], use_cache: bool) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        return self._transfuser_model(features, use_cache)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return transfuser_loss(targets, predictions, self._config)

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        optimizer = torch.optim.Adam(self._transfuser_model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/score_epoch"
            }
        }

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        best_model_checkpoint = ModelCheckpoint(
            monitor='val/score_epoch',               
            mode='max',                   
            save_top_k=3,                      
            filename='best_{epoch}_{val/score_epoch:.4f}',  
            save_weights_only=False            
        )
        every_10_epoch_checkpoint = ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=10,
            filename='{epoch}_{step}',
            save_weights_only=False
        )

        return [TransfuserCallback(self._config), best_model_checkpoint, every_10_epoch_checkpoint]
