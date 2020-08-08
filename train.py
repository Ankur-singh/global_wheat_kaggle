import ast
import torch
import numpy as np
import pandas as pd
from functools import partial
from collections import OrderedDict
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from utils import load_obj, send_message, printm
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core.lightning import LightningModule as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

# custom
from dataset import DatasetRetriever, get_train_transforms, get_valid_transforms, collate_fn


def get_net(pimg_sz, cp=None, img_sz=None):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=True)

    config.num_classes = 1
    config.image_size = pimg_sz
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    if cp and img_sz:
        checkpoint = torch.load(cp)

        state_dict = OrderedDict()
        for k in checkpoint['state_dict'].keys():
            splits = k.split('.')
            new_k = '.'.join(splits[2:])
            if new_k != 'anchors.boxes':
                state_dict[new_k] = checkpoint['state_dict'][k]

        net.load_state_dict(state_dict)
        
        config.num_classes = 1
        config.image_size = img_sz
        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    
    return DetBenchTrain(net, config)


class Net(pl):
    def __init__(self, model, config):
        super(Net, self).__init__()
        self.config = config
        self.model = model
        self.hparams = self.config

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        images = torch.stack(images)
        images = images.float()
        boxes = [target['boxes'] .float() for target in targets]
        labels = [target['labels'].float() for target in targets]

        loss, _, _ = self.model(images, boxes, labels)
        return {'loss': loss}

    def training_epoch_end(self, train_output):
        train_epoch_loss = torch.stack([x['loss'] for x in train_output]).mean()
        if self.config.notify:
            send_message(f'Your `train_loss` is {train_epoch_loss}')

        return {'train_loss': train_epoch_loss,
                'log': {'train_loss': train_epoch_loss}}

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        images = torch.stack(images)
        images = images.float()
        boxes = [target['boxes'] .float() for target in targets]
        labels = [target['labels'].float() for target in targets]

        loss, _, _ = self.model(images, boxes, labels)
        return {'loss': loss}

    def validation_epoch_end(self, val_output):
        val_epoch_loss = torch.stack([x['loss'] for x in val_output]).mean()
        if self.config.notify:
            send_message(f'Your `val_loss` is {val_epoch_loss}')

        return {'val_loss': val_epoch_loss,
                'log': {'val_loss': val_epoch_loss}}

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
                                 'weight_decay': 0.001},
                                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                                 'weight_decay': 0.0}]

        optimizer = load_obj(self.config.optimizer.class_name)(optimizer_parameters, **self.config.optimizer.params)
        if self.config.scheduler.steps_per_epoch:
            scheduler = load_obj(self.config.scheduler.class_name)(optimizer, 
                                                                    steps_per_epoch=len(self.train_dataset)//(self.config.data.batch_size), 
                                                                    **self.config.scheduler.params)
        else:
            scheduler = load_obj(self.config.scheduler.class_name)(optimizer, **self.config.scheduler.params)
            
        return [optimizer], [{"scheduler": scheduler,
                              "interval": self.config.scheduler.step,
                              "monitor": self.config.scheduler.monitor}]

    def prepare_data(self):
        df_folds = pd.read_csv(self.config.data.train_folds)
        markings = pd.read_csv(self.config.data.train_ext)
        bboxs = np.stack(markings['bbox'].apply(lambda x: ast.literal_eval(x)))
        for i, column in enumerate(['x', 'y', 'w', 'h']):
            markings[column] = bboxs[:, i]
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
                                  num_workers=self.config.data.num_workers, 
                                  collate_fn=collate_fn)

        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.valid_dataset, 
                                batch_size=self.config.data.batch_size,
                                pin_memory=False, shuffle=False,
                                num_workers=self.config.data.num_workers, 
                                collate_fn=collate_fn)

        return val_loader


if __name__ == "__main__":

    from omegaconf import OmegaConf
    conf = OmegaConf.load('config.yaml')
    cli_conf = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf, cli_conf)

    if conf.notify:
        assert conf.bot_token, "If you want to get notified, you should specify your bot_token"
        assert conf.chat_id, "If you want to get notified, you should specify your chat_id"
        send_message = partial(send_message, name=conf.name, chat_id=conf.chat_id, bot_token=conf.bot_token)

    if conf.weights and not conf.resume:
        if conf.data.pimg_sz is None: conf.data.pimg_sz = conf.data.img_sz
        model = get_net(conf.data.pimg_sz, conf.weights, conf.data.img_sz)
    else:
        model = get_net(conf.data.img_sz)
    
    net = Net(model, conf)

    if conf.resume:
        trainer = Trainer(resume_from_checkpoint=conf.resume)
    else:
        checkpoint_callback = ModelCheckpoint(filepath=f'{conf.cp_path}'+'/{epoch}-{val_loss:.2f}')
        early_stopping = EarlyStopping('val_loss')
        tb_logger = TensorBoardLogger(conf.logs_path)

        trainer = Trainer(logger=[tb_logger],
                          early_stop_callback=early_stopping,
                          checkpoint_callback=checkpoint_callback,
                          **conf.trainer)

    if conf.trainer.auto_lr_find:
        lr_finder = trainer.lr_find(net)
        new_lr = lr_finder.suggestion()
        net.optimizer.params.lr = new_lr

    printm()

    trainer.fit(net)
