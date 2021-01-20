from keras.optimizers import Adadelta
import imgaug.augmenters as iaa

from tensorflow.python.keras.callbacks import CSVLogger, ModelCheckpoint

from src.experiments.data_loader import DataGenerator
from src.experiments.sim2real.nn import nn
from src.experiments import ConfusionMatrixPlotter
from src.experiments.sim2real.util.acc_loss_cb import AccLossPlotter

import os
import cv2
import numpy as np

from src.experiments.model import smartlab_gelsight2014_config
from src.experiments.model import SimulationApproach
from random import randint

smartlab_gelsight2014_config['background_img'] = cv2.resize(cv2.imread('aligned/background.png'), dsize=(224, 224))

rnd_bkgs = []
n_rnd_bkgs = 12
for i in range(n_rnd_bkgs):
    rnd_bkgs.append(
        cv2.resize(cv2.imread('textures/' + str(i + 1) + '.png', cv2.IMREAD_GRAYSCALE), dsize=(224, 224)) / 225.0)

simulation2014_approach = SimulationApproach(**smartlab_gelsight2014_config)


def texture_augmented_augmentor(x, y):
    def config_and_generate(depth_i, cls_i):
        rand_idx = randint(0, n_rnd_bkgs - 1)
        depth_seq = iaa.Sequential([
            iaa.Affine(scale=(0.5, 1.5), rotate=(-45, 45))
        ])
        depth_i += depth_seq(images=np.array([rnd_bkgs[rand_idx]]))[0] * 0.1
        return simulation2014_approach.generate(depth_i)

    sim = np.array([config_and_generate(x[i], y[i]) for i in range(len(x))])

    seq = iaa.Sequential([
        iaa.RandAugment(n=2, m=9)
    ])

    return seq(images=sim), y


def augmentor(x, y):
    seq = iaa.Sequential([
        iaa.RandAugment(n=2, m=9)
    ])

    return seq(images=x), y


train_real_generator = DataGenerator(
    path='aligned/real/',
    # path='aligned/depth/',
    splits_file='aligned/index.yaml',
    output_paths=False,
    split='train',
    augmentor=augmentor,
    # augmentor=texture_augmented_augmentor
)

train_sim_generator = DataGenerator(
    path='aligned/sim/',
    splits_file='aligned/index.yaml',
    output_paths=False,
    split='train',
    augmentor=augmentor
)

validation_real_generator = DataGenerator(
    path='aligned/real/',
    splits_file='aligned/index.yaml',
    output_paths=False,
    split='validation',
    batch_size=1
)

validation_sim_generator = DataGenerator(
    path='aligned/sim/',
    splits_file='aligned/index.yaml',
    output_paths=False,
    split='validation',
    batch_size=1
)

epochs = 300
steps_per_epoch = 2
n_classes = 21
fc_d = 128
optimizer = Adadelta(0.1)
exp_name = 'sim2real_300it_128fc_lr0x1'
train_generator = train_sim_generator
validation_generator = validation_real_generator

model = nn(n_classes, fc_d=fc_d)

model.compile(
    optimizer=optimizer,
    loss=['categorical_crossentropy'],
    metrics=['accuracy']
)

logs_path = 'logs/' + exp_name + '/'
os.mkdir(logs_path)
csv_logger = CSVLogger(logs_path + 'log.log')
model_checkpoint = ModelCheckpoint(
    filepath=logs_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

x, y = next(validation_generator)
model.fit(train_generator,
          validation_data=validation_generator,
          validation_batch_size=1,
          validation_steps=32,
          steps_per_epoch=steps_per_epoch,
          epochs=epochs,
          callbacks=[
              csv_logger,
              model_checkpoint,
              ConfusionMatrixPlotter(x, y, validation_generator.classes,
                                     to_file=logs_path + 'confusion_matrix.png'),
              AccLossPlotter(to_file=logs_path + 'plot_history.png')
          ])

