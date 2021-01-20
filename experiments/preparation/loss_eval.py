from ..data_loader import data_generator, DataGenerator
from ..fine_tuning.losses import rectified_mae_loss, psnr_loss, ssim_loss

import numpy as np

generator = DataGenerator(
    'aligned/real',
    sim_path='aligned/legacy',
    shuffle=False,
    batch_size=1,
    resize=False)

maes = []
psnrs = []
ssims = []

n = 0

print('GENERATOR SIZE: ', generator.size())
for (real, sim, cls, path) in generator:
    real = np.array(real)
    sim = np.array(sim)

    c_mae = rectified_mae_loss(real, sim)
    c_psnr = psnr_loss(real, sim)
    c_ssim = ssim_loss(real, sim)

    maes.append(c_mae)
    psnrs.append(c_psnr)
    ssims.append(c_ssim)

    n += 1

    if len(maes) % 50 == 0:
        print(str(int(n/generator.size() * 100)) + '%')



print('mean MAE:', np.mean(maes), np.var(maes))
print('mean PSNR:', np.mean(psnrs), np.var(psnrs))
print('mean SSIM:', np.mean(ssims), np.var(ssims))
