from bayes_opt import BayesianOptimization
import numpy as np

import cv2
import matplotlib.pyplot as plt

from .losses import ssim_loss
from ..data_loader import data_generator, load_single_img2, DataGenerator
from src.experiments.model import SimulationApproach
from ..model.configs import mit_gelsight2017_config, smartlab_gelsight2014_config

pbounds = {
    'kernel_size': (3, 51),
    't': (1, 10),
    'sigma': (1, 10)
    # 'l1_r': (0, 255),
    # 'l1_g': (0, 255),
    # 'l1_b': (0, 255),
    # 'l2_r': (0, 255),
    # 'l2_g': (0, 255),
    # 'l2_b': (0, 255),
    # 'l3_r': (0, 255),
    # 'l3_g': (0, 255),
    # 'l3_b': (0, 255),
    # 'l4_r': (0, 255),
    # 'l4_g': (0, 255),
    # 'l4_b': (0, 255),
    # 'texture_sigma': (0.0000001, 0.00001),
    # 'ka': (0, 1)
}

generator = DataGenerator(
    'aligned/real',
    depth_path='aligned/depth',
    shuffle=True,
    batch_size=32)

def plot_loss(res_tmp):
    plt.clf()

    plt.plot(res_tmp)
    plt.savefig('progress.png')


bkgs = {}
for cls in generator.classes:
    bkgs[cls] = load_single_img2(
        'aligned/background_' + cls + '.png')

res_tmp = []


def sample_and_generate(
        # l1_r, l1_g, l1_b,
        # l2_r, l2_g, l2_b,
        # l3_r, l3_g, l3_b,
        # l4_r, l4_g, l4_b,
        # texture_sigma, ka
        t, sigma, kernel_size
):
    real, depth, cls, _ = next(generator)

    def config_and_generate(depth_i, cls_i):
        cls_i_name = generator.classes[np.argmax(cls_i)]
        smartlab_gelsight2014_config['background_img'] = bkgs[cls_i_name]
        simulation2014_approach = SimulationApproach(
            **{
                **smartlab_gelsight2014_config,
                **{
                    'sigma': int(sigma),
                    't': int(t),
                    'kernel_size': int(kernel_size),
                    # 'light_sources': [
                    #     {'position': [0, 1, 0.25], 'color': (l1_r, l1_g, l1_b)},
                    #     {'position': [-1, 0, 0.25], 'color': (l2_r, l2_g, l2_b)},
                    #     {'position': [0, -1, 0.25], 'color': (l3_r, l3_g, l3_b)},
                    #     {'position': [1, 0, 0.25], 'color': (l4_r, l4_g, l4_b)},
                    # ],
                    # 'ka': ka,
                    # 'texture_sigma': texture_sigma
                }
            }
        )
        return simulation2014_approach.generate(depth_i)

    sim = np.array([config_and_generate(depth[i], cls[i]) for i in range(len(depth))])
    loss = ssim_loss(sim, real)

    res_tmp.append(loss)
    plot_loss(res_tmp)

    cv2.imshow('preview', np.concatenate([sim[0], sim[1], sim[2], sim[3], sim[4]], axis=1))
    cv2.waitKey(1)

    return loss


optimizer = BayesianOptimization(
    f=sample_and_generate,
    pbounds=pbounds,
    verbose=2,
    random_state=25
)

optimizer.maximize(
    init_points=5,
    n_iter=1000
)

import matplotlib.pyplot as plt

plt.plot([res['target'] for i, res in enumerate(optimizer.res)])
plt.ylabel('some numbers')
plt.show()
