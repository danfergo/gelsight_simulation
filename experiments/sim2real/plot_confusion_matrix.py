from src.experiments.sim2real.nn import nn
from src.experiments.data_loader import DataGenerator, OBJECT_SET_CLASSES
from src.experiments import ConfusionMatrixPlotter

import numpy as np
n_classes = 21

model = nn(n_classes, fc_d=128)
exp_name = 'sim2real'
logs_path = 'logs/' + exp_name + '/'

validation_sim_generator = DataGenerator(
    path='aligned/real/',
    splits_file='aligned/index.yaml',
    output_paths=False,
    split='validation',
    batch_size=-1,
)

model.compile()

model.load_weights(logs_path)

x, y = next(validation_sim_generator)
cm = ConfusionMatrixPlotter(x, y, OBJECT_SET_CLASSES, interactive=False)
cm.model = model
cm.on_epoch_end()
