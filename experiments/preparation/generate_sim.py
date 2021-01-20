from ..data_loader import data_generator, DataGenerator
from ..model.configs import smartlab_gelsight2014_config
from ..model.sim_model import SimulationApproach, SimulationApproachLegacy

import cv2
import numpy as np

generator = DataGenerator(
    '/aligned/real',
    depth_path='aligned/depth',
    shuffle=False,
    batch_size=1,
    resize=False)

# smartlab_gelsight2014_config['background_img'] = cv2.imread('aligned/background.png')

for (_, depth, cls, path) in generator:
    cls_name = generator.classes[np.argmax(cls)]

    smartlab_gelsight2014_config['background_img'] = cv2.imread('aligned/background_' + cls_name + '.png')

    simulation2014_approach = SimulationApproachLegacy(
        **smartlab_gelsight2014_config
    )

    sim = simulation2014_approach.generate(depth[0])
    cv2.imwrite('aligned/' + path[0], sim)
    cv2.imshow('sim', sim)
    cv2.waitKey(1)

    print(path)
