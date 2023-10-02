from typing import Dict, Any

import tensorflow as tf
from dataclasses import dataclass, field
import wandb
from omegaconf import DictConfig, OmegaConf
import os

from .utils import make_image_grid
from datasets import store as data_store
from hydra.utils import instantiate
from backbones.store import build_backbone
from training.opt_store import build_optimizer
import logging
import sys


class Trainer(object):
    def __init__(
        self,
        experiment_root_dir: str,
        model_dir: str,
        log_dir: str,
        max_checkpoints_to_keep: int,
        checkpoint_name: str,
        num_classes: int,
        input_height: int,
        input_width: int,
        backbone: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        dataset_train: tf.data.Dataset,
        dataset_validation: tf.data.Dataset,
        log_to_wandb: bool,
        wandb_config_dict: Dict[str, Any] = {},
        training_epochs: int = 150,
        log_steps: int = 500,
        restore_from_epoch: int | None = None,
    ):
        self.experiment_root_dir = experiment_root_dir
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.max_checkpoints_to_keep = max_checkpoints_to_keep
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.backbone = backbone
        self.optimizer = optimizer
        self.dataset_train = dataset_train
        self.dataset_validation = dataset_validation
        self.log_to_wandb = log_to_wandb
        self.wandb_config_dict = wandb_config_dict
        self.training_epochs = training_epochs
        self.log_steps = log_steps
        self.restore_from_epoch = restore_from_epoch
        self.top1_train = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
        self.top5_train = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
        self.top1_val = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
        self.top5_val = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
        self.avg_ce_train = tf.keras.metrics.Mean()
        self.avg_reg_train = tf.keras.metrics.Mean()
        self.avg_total_train = tf.keras.metrics.Mean()
        self.avg_total_val = tf.keras.metrics.Mean()
        self.avg_reg_val = tf.keras.metrics.Mean()
        self.avg_ce_val = tf.keras.metrics.Mean()
        self.training_step_count = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int64, name="training_step_count"
        )
        self.epochs_trained = tf.Variable(
            initial_value=0, trainable=False, dtype=tf.int64, name="epochs_completed"
        )

        self.checkpoint = tf.train.Checkpoint(
            backbone=self.backbone,
            optimizer=self.optimizer,
            epochs_trained=self.epochs_trained,
            training_step_count=self.training_step_count,
            top1_train=self.top1_train,
            top5_train=self.top5_train,
            top1_val=self.top1_val,
            top5_val=self.top5_val,
        )

        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint,
            directory=self.model_dir,
            max_to_keep=self.max_checkpoints_to_keep,
            checkpoint_name=self.checkpoint_name,
        )

        if self.log_to_wandb:
            wandb.init(**self.wandb_config_dict)
            wandb.define_metric("training_steps")
            wandb.define_metric("epochs_trained")
            wandb.define_metric(
                "train/step_progress/ce_loss", step_metric="training_steps"
            )
            wandb.define_metric(
                "train/step_progress/reg_loss", step_metric="training_steps"
            )
            wandb.define_metric(
                "train/step_progress/total_loss", step_metric="training_steps"
            )
            wandb.define_metric(
                "train/step_progress/top1_accuracy",
                step_metric="training_steps",
            )
            wandb.define_metric(
                "train/step_progress/top5_accuracy",
                step_metric="training_steps",
            )
            wandb.define_metric(
                "train/step_progress/input_image_grid",
                step_metric="training_steps",
            )

            wandb.define_metric(
                "train/epoch_progress/ce_loss", step_metric="epochs_trained"
            )
            wandb.define_metric(
                "train/epoch_progress/reg_loss", step_metric="epochs_trained"
            )
            wandb.define_metric(
                "train/epoch_progress/total_loss", step_metric="epochs_trained"
            )
            wandb.define_metric(
                "train/epoch_progress/top1_accuracy",
                step_metric="epochs_trained",
            )
            wandb.define_metric(
                "train/epoch_progress/top5_accuracy",
                step_metric="epochs_trained",
            )

            wandb.define_metric(
                "val/epoch_progress/ce_loss", step_metric="epochs_trained"
            )
            wandb.define_metric(
                "val/epoch_progress/reg_loss", step_metric="epochs_trained"
            )
            wandb.define_metric(
                "val/epoch_progress/total_loss", step_metric="epochs_trained"
            )
            wandb.define_metric(
                "val/epoch_progress/top1_accuracy",
                step_metric="epochs_trained",
            )
            wandb.define_metric(
                "val/epoch_progress/top5_accuracy",
                step_metric="epochs_trained",
            )

        self.logger_train_step = logging.getLogger("train_step")
        self.logger_train_epoch = logging.getLogger("train_epoch")
        self.logger_val_epoch = logging.getLogger("val_epoch")

        self.logger_train_step.addHandler(
            logging.FileHandler(
                filename=os.path.join(self.log_dir, "train_step.log"), mode="a"
            )
        )

        self.logger_train_epoch.addHandler(
            logging.FileHandler(
                filename=os.path.join(self.log_dir, "train_epoch.log"), mode="a"
            )
        )

        self.logger_val_epoch.addHandler(
            logging.FileHandler(
                filename=os.path.join(self.log_dir, "val_epoch.log"), mode="a"
            )
        )

    @tf.function(jit_compile=True)
    def train_step(self, images: tf.Tensor, labels: tf.Tensor):
        with tf.GradientTape() as tape:
            logits = self.backbone(images, training=True)
            loss_ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
                labels, logits
            )
            reg_loss = tf.reduce_sum(self.backbone.losses)
            total_loss = loss_ce + reg_loss
        grads = tape.gradient(total_loss, self.backbone.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.backbone.trainable_weights))
        self.top1_train.update_state(labels, logits)
        self.top5_train.update_state(labels, logits)
        self.avg_ce_train.update_state(loss_ce)
        self.avg_reg_train.update_state(reg_loss)
        self.avg_total_train.update_state(total_loss)
        self.training_step_count.assign_add(1)
        return loss_ce, reg_loss, total_loss

    @tf.function(jit_compile=True)
    def validation_step(self, images: tf.Tensor, labels: tf.Tensor):
        logits = self.backbone(images, training=False)
        loss_ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(
            labels, logits
        )
        reg_loss = tf.reduce_sum(self.backbone.losses)
        total_loss = loss_ce + reg_loss
        self.top1_val.update_state(labels, logits)
        self.top5_val.update_state(labels, logits)
        self.avg_ce_val.update_state(loss_ce)
        self.avg_reg_val.update_state(reg_loss)
        self.avg_total_val.update_state(total_loss)
        return loss_ce, reg_loss, total_loss

    def run(self):
        if self.restore_from_epoch is None:
            if self.ckpt_manager.latest_checkpoint is None:
                self.ckpt_manager.restore_or_initialize()
            else:
                self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
        else:
            model_name = (
                f"{self.model_dir}/{self.checkpoint_name}-{self.restore_from_epoch}"
            )
            self.checkpoint.restore(model_name)

        self.backbone.summary()
        for CURRENT_EPOCH in range(self.epochs_trained.numpy(), self.training_epochs):
            for data in self.dataset_train:
                # data = (images, labels)
                loss_tuple = self.train_step(data[0], data[1])

                if self.training_step_count.numpy() % self.log_steps == 0:
                    if self.log_to_wandb:
                        img_grid = wandb.Image(
                            data_or_path=make_image_grid(data[0].numpy()),
                            caption=f"Input image grid",
                        )
                        log_dict = {
                            "train/step_progress/ce_loss": loss_tuple[0],
                            "train/step_progress/reg_loss": loss_tuple[1],
                            "train/step_progress/total_loss": loss_tuple[2],
                            "train/step_progress/top1_accuracy": self.top1_train.result(),
                            "train/step_progress/top5_accuracy": self.top5_train.result(),
                            "train/step_progress/input_image_grid": img_grid,
                            "training_steps": self.training_step_count,
                        }

                        wandb.log(log_dict)
                    self.logger_train_step.info(
                        f"Epoch : {CURRENT_EPOCH + 1}/{self.training_epochs},\n "
                        f"training steps : {self.training_step_count.numpy()}, "
                        f"Average CE loss : {loss_tuple[0].numpy():.4f}, "
                        f"Regularization loss : {loss_tuple[1].numpy():.4f}, "
                        f"Total Loss : {loss_tuple[2].numpy():.4f}, "
                        f"Top-1 acc : {self.top1_train.result().numpy():.4f}, "
                        f"Top-5 acc : {self.top5_train.result().numpy():.4f}."
                    )

            self.epochs_trained.assign_add(1)
            log_dict = {
                "train/epoch_progress/ce_loss": self.avg_ce_train.result(),
                "train/epoch_progress/reg_loss": self.avg_reg_train.result(),
                "train/epoch_progress/total_loss": self.avg_total_train.result(),
                "train/epoch_progress/top1_accuracy": self.top1_train.result(),
                "train/epoch_progress/top5_accuracy": self.top5_train.result(),
                "epochs_trained": self.epochs_trained,
            }

            wandb.log(log_dict)
            self.logger_train_epoch.info(
                f"Training set Summary statistics after epoch {self.epochs_trained.numpy()}"
            )
            self.logger_train_epoch.info(
                f"Epoch : {self.epochs_trained.numpy()}/{self.training_epochs},\n training steps : {self.training_step_count.numpy()}, Average CE loss : {self.avg_ce_train.result().numpy():.4f}, Regularization loss : {self.avg_reg_train.result().numpy():.4f}, Total Loss : {self.avg_total_train.result().numpy():.4f}, Top-1 acc : {self.top1_train.result().numpy():.4f}, Top-5 acc : {self.top5_train.result().numpy():.4f}."
            )

            self.ckpt_manager.save(checkpoint_number=CURRENT_EPOCH)
            self.top1_train.reset_state()
            self.top5_train.reset_state()
            self.avg_ce_train.reset_state()
            self.avg_total_train.reset_state()
            self.avg_reg_train.reset_state()
            for data in self.dataset_validation:
                _ = self.validation_step(data[0], data[1])

            log_dict = {
                "val/epoch_progress/ce_loss": self.avg_ce_val.result(),
                "val/epoch_progress/reg_loss": self.avg_reg_val.result(),
                "val/epoch_progress/total_loss": self.avg_total_val.result(),
                "val/epoch_progress/top1_accuracy": self.top1_val.result(),
                "val/epoch_progress/top5_accuracy": self.top5_val.result(),
                "epochs_trained": self.epochs_trained,
            }

            wandb.log(log_dict)
            self.logger_val_epoch.info(
                f"Validation set summary statistics after epoch {self.epochs_trained.numpy()}."
            )
            self.logger_val_epoch.info(
                f"Epoch : {self.epochs_trained.numpy()}/{self.training_epochs},\n "
                f"training steps : {self.training_step_count.numpy()}, "
                f"Average CE loss : {self.avg_ce_val.result().numpy():.4f}, "
                f"Regularization loss : {self.avg_reg_val.result().numpy():.4f}, "
                f"Total Loss : {self.avg_total_val.result().numpy():.4f}, "
                f"Top-1 acc : {self.top1_val.result().numpy():.4f}, "
                f"Top-5 acc : {self.top5_val.result().numpy():.4f}."
            )
            self.avg_ce_train.reset_state()
            self.avg_reg_val.reset_state()
            self.avg_total_val.reset_state()
            self.top1_val.reset_state()
            self.top5_val.reset_state()


def build_trainer_from_config(cfg: DictConfig) -> Trainer:
    experiment_root_dir = cfg.experiment_root_dir
    os.makedirs(experiment_root_dir, exist_ok=True)
    model_dir = cfg.model_dir
    os.makedirs(model_dir, exist_ok=True)
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(log_dir, f"{cfg.checkpoint_name}.yaml"))

    preprocessing_train = instantiate(cfg.preprocessing_training, _convert_="object")
    train_db_config = cfg.training_dataset
    train_db = data_store.get_dataset(
        train_db_config, split="train", transforms=preprocessing_train
    )

    preprocessing_val = instantiate(cfg.preprocessing_val, _convert_="object")

    val_db_config = cfg.val_dataset
    val_db = data_store.get_dataset(
        val_db_config, split="validation", transforms=preprocessing_val
    )

    backbone_config = cfg.backbone
    input_height = cfg.input_height
    input_width = cfg.input_width
    num_classes = cfg.num_classes
    input_shape = (input_height, input_width, 3)
    backbone = build_backbone(
        cfg=backbone_config, input_shape=input_shape, num_classes=num_classes
    )
    optimizer_config = cfg.optimizer
    optimizer = build_optimizer(optimizer_config)
    log_steps = cfg.log_steps
    restore_from_epoch = cfg.restore_from_epoch

    training_epochs = cfg.training_epochs
    max_checkpoints_to_keep = cfg.max_checkpoints_to_keep
    wandb_config = cfg.wandb
    log_to_wandb = cfg.log_to_wandb
    checkpoint_name = cfg.checkpoint_name
    trainer = Trainer(
        experiment_root_dir=experiment_root_dir,
        model_dir=model_dir,
        log_dir=log_dir,
        max_checkpoints_to_keep=max_checkpoints_to_keep,
        checkpoint_name=checkpoint_name,
        num_classes=num_classes,
        input_height=input_height,
        input_width=input_width,
        backbone=backbone,
        optimizer=optimizer,
        dataset_train=train_db,
        dataset_validation=val_db,
        log_to_wandb=log_to_wandb,
        wandb_config_dict=wandb_config,
        training_epochs=training_epochs,
        log_steps=log_steps,
        restore_from_epoch=restore_from_epoch,
    )
    return trainer
