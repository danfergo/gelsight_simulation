from src.experiments.sim2real.nn import nn
from src.experiments.data_loader import DataGenerator, OBJECT_SET_CLASSES, to_categorical
from src.experiments import ConfusionMatrixPlotter

import numpy as np
from sklearn.metrics import accuracy_score

validation_sim_generator = DataGenerator(
    path='aligned/real/',
    splits_file='/aligned/index.yaml',
    output_paths=False,
    split='test',
    batch_size=-1,
)

n_classes = 21

model = nn(n_classes, fc_d=128)
exp_name = 'dynamic_train_sim2real_300it_128fc_lr0x1'
logs_path = 'logs/' + exp_name + '/'

model.compile()

model.load_weights(logs_path)

x, y = next(validation_sim_generator)

cm = ConfusionMatrixPlotter(x, y, OBJECT_SET_CLASSES, interactive=False)
cm.model = model
cm.on_epoch_end()

pred = model.predict(x)

max_pred = np.argmax(pred, axis=1)
max_y = np.argmax(y, axis=1)
score = accuracy_score(max_y, max_pred)

print('Accuracy: ', score)
print('N classes: ', len(OBJECT_SET_CLASSES))
