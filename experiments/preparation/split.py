from itertools import groupby
from random import sample
from yaml import dump

from ..data_loader import DataGenerator

generator = DataGenerator(
    'aligned/real/',
    shuffle=False,
    batch_size=1,
)

splits = {
    'train': [],
    'validation': [],
    'test': []
}

for k, g in groupby(generator.files, lambda f: generator.filename_data(f)[0]):
    files = list(g)

    train_split = sample(files, 70)
    validation_split = sample([x for x in files if x not in train_split], 20)
    test_split = sample([x for x in files if x not in train_split + validation_split], 8)

    splits['train'] += train_split
    splits['validation'] += validation_split
    splits['test'] += test_split

stream = open('aligned/index.yaml', 'w')
dump(splits, stream)