import csv
import matplotlib.pyplot as plt
import argparse

def parse(epoch, accuracy, loss, val_accuracy, val_loss):
    print(epoch)
    epoch = [int(i) for i in epoch]
    accuracy = [float(i) for i in accuracy]
    loss = [float(i) for i in loss]
    val_accuracy = [float(i) for i in val_accuracy]
    val_loss = [float(i) for i in val_loss]
    return epoch, accuracy, loss, val_accuracy, val_loss


def read_log(filename):
    return parse(*list(map(list, zip(*list(csv.reader(open(filename, 'r')))[1:]))))


from scipy.ndimage.filters import gaussian_filter1d

max_epochs = 300

fig, (ax1, ax2_) = plt.subplots(1, 2)
ax2 = ax1.twinx()
ax2.tick_params(labelright='off')

epoch, accuracy, loss, val_accuracy, val_loss = read_log('logs/sim2sim_300it_128fc_lr0x1/log.log')
epoch = epoch[0:max_epochs]
accuracy, loss, val_accuracy, val_loss = gaussian_filter1d(accuracy, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(loss, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(val_accuracy, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(val_loss, sigma=4)[0:max_epochs]
ax1.plot(epoch, val_accuracy, label='Sim2Sim')
ax2.plot(epoch, val_loss, alpha=0.3)

# ==================
epoch, accuracy, loss, val_accuracy, val_loss = read_log('logs/sim2real_300it_128fc_lr0x1/log.log')
epoch = epoch[0:max_epochs]
accuracy, loss, val_accuracy, val_loss = gaussian_filter1d(accuracy, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(loss, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(val_accuracy, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(val_loss, sigma=4)[0:max_epochs]
ax1.plot(epoch, val_accuracy, label='Sim2Real')
ax2.plot(epoch, val_loss, alpha=0.3)
# ==================

epoch, accuracy, loss, val_accuracy, val_loss = read_log('logs/dynamic_train_sim2real_300it_128fc_lr0x1/log.log')
epoch = epoch[0:max_epochs]
accuracy, loss, val_accuracy, val_loss = gaussian_filter1d(accuracy, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(loss, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(val_accuracy, sigma=4)[0:max_epochs], \
                                         gaussian_filter1d(val_loss, sigma=4)[0:max_epochs]
ax1.plot(epoch, val_accuracy, label='Sim2Real (texture augmented)')
ax2.plot(epoch, val_loss, alpha=0.3)


# ==================

ax1.set_title('Sim2Real learning')
ax1.set_ylabel('Accuracy')
ax2.set_ylabel('Loss')
ax1.set_xlabel('Epochs')

ax1.legend(loc='upper left')

plt.show()
