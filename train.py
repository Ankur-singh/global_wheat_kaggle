import ast
import torch
import importlib
import numpy as np
import pandas as pd
from typing import Any
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core.lightning import LightningModule as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

## custom
from dataset import DatasetRetriever, get_train_transforms, get_valid_transforms, collate_fn

def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            "Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path)
        )
    return getattr(module_obj, obj_name)

def get_net(img_sz):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=True)

    config.num_classes = 1
    config.image_size = img_sz
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    
    return DetBenchTrain(net, config)

class Net(pl):
    def __init__(self, model, config):
      super(Net, self).__init__()
      self.config = config
      self.model = model

    def forward(self, x, *args, **kwargs):
      return self.model(x)

    def training_step(self, batch, batch_idx):
      images, targets, image_ids = batch
      images = torch.stack(images)
      images = images.float()
      boxes  = [target['boxes'] .float() for target in targets]
      labels = [target['labels'].float() for target in targets]

      loss, _, _ = self.model(images, boxes, labels)
      return {'loss': loss}

    def training_epoch_end(self, train_output):
        train_epoch_loss = torch.stack([x['loss'] for x in train_output]).mean()
        return {'train_loss': train_epoch_loss,
                'log': {'train_loss': train_epoch_loss}}

    def validation_step(self, batch, batch_idx):
      images, targets, image_ids = batch
      images = torch.stack(images)
      images = images.float()
      boxes  = [target['boxes'] .float() for target in targets]
      labels = [target['labels'].float() for target in targets]

      loss, _, _ = self.model(images, boxes, labels)
      return {'loss': loss}

    def validation_epoch_end(self, val_output):
        val_epoch_loss = torch.stack([x['loss'] for x in val_output]).mean()
        return {'val_loss': val_epoch_loss,
                'log': {'val_loss': val_epoch_loss}}

    def configure_optimizers(self):
      optimizer = load_obj(self.config.optimizer.class_name)(self.model.parameters(), **self.config.optimizer.params)
      scheduler = load_obj(self.config.scheduler.class_name)(optimizer, **self.config.scheduler.params)
      return [optimizer], [{"scheduler": scheduler,
                            "interval" : self.config.scheduler.step,
                            "monitor"  : self.config.scheduler.monitor}]

    def prepare_data(self):
      df_folds = pd.read_csv(self.config.data.train_folds)
      markings = pd.read_csv(self.config.data.train_ext)
      bboxs = np.stack(markings['bbox'].apply(lambda x: ast.literal_eval(x)))
      for i, column in enumerate(['x', 'y', 'w', 'h']):
          markings[column] = bboxs[:,i]
      markings.drop(columns=['bbox'], inplace=True)

      self.train_dataset = DatasetRetriever(image_ids=df_folds[df_folds['fold'] != self.config.data.fold].image_id.values, 
                                       path=self.config.data.path,
                                       marking=markings, 
                                       transforms=get_train_transforms(self.config.data.img_sz), 
                                       test=False)
      
      
      self.valid_dataset = DatasetRetriever(image_ids=df_folds[df_folds['fold'] == self.config.data.fold].image_id.values, 
                                       path=self.config.data.path,
                                       marking=markings, 
                                       transforms=get_valid_transforms(self.config.data.img_sz), 
                                       test=True)

    def train_dataloader(self):
      train_loader = DataLoader(self.train_dataset,
                              batch_size=self.config.data.batch_size,
                              pin_memory=False, drop_last=True,
                              num_workers=self.config.data.num_workers, collate_fn=collate_fn)
      
      return train_loader

    def val_dataloader(self):
      val_loader = DataLoader(self.valid_dataset, 
                            batch_size=self.config.data.batch_size,
                            pin_memory=False, shuffle=False,
                            num_workers=self.config.data.num_workers, collate_fn=collate_fn)   
      
      return val_loader

if __name__ == "__main__":

    from omegaconf import OmegaConf
    conf = OmegaConf.load('config.yaml')
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    model = get_net(conf.data.img_sz)
    net = Net(model, conf)

    if conf.pretrained:
        trainer = Trainer(resume_from_checkpoint=conf.pretrained)
    else:
        checkpoint_callback = ModelCheckpoint(filepath=f'{conf.cp_path}'+'/{epoch}-{val_loss:.2f}')
        early_stopping = EarlyStopping('val_loss')
        tb_logger = TensorBoardLogger(conf.logs_path)

        trainer = Trainer(logger = [tb_logger],
                        early_stop_callback = early_stopping,
                        checkpoint_callback = checkpoint_callback,
                        **conf.trainer)

    trainer.fit(net)

