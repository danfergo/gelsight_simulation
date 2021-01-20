from random import random
import matplotlib.pyplot as plt

from src.experiments.data_loader import DataGenerator, load_single_img2
from src.experiments.fine_tuning.losses import ssim_loss
from src.experiments.model import smartlab_gelsight2014_config

import numpy as np
import cv2

from src.experiments.model import SimulationApproach


def sample():
    return {
        'sigma': int(1 + 10 * random()),
        'kernel_size': int(5 + 25 * random()),
        't': int(1 + 7)
    }


generator = DataGenerator(
    'aligned/real',
    depth_path='aligned/depth',
    shuffle=True,
    batch_size=32)


def plot_loss(res_tmp):
    plt.clf()

    plt.plot(res_tmp)
    plt.savefig('progress_deformation_random.png')

bkgs = {}
for cls in generator.classes:
    bkgs[cls] = load_single_img2('aligned/background_' + cls + '.png')

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
                    # 'sigma': int(sigma),
                    # 't': int(t),
                    # 'kernel_size': int(kernel_size),
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


n_epochs = 1000

params = sample()
e = 0
print("|\t".join([''] + [k for k in params] + ['LOSS']))
for i in range(n_epochs):
    params = sample()
    loss = sample_and_generate(**params)
    e += 0
    print("|\t".join([str(i)] + [str(round(params[k], 2)) for k in params] + [str(round(loss, 3))]))

