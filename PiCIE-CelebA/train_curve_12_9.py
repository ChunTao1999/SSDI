import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=16)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes
plt.rc('lines', lw=2)

epochs = np.arange(10)
kmeans_loss_view1 = [0.54694, 0.25877, 0.15294, 0.10242, 0.09252, 0.06916, 0.06334, 0.06415, 0.05675, 0.04960]
kmeans_loss_view2 = [0.52149, 0.25050, 0.14507, 0.10626, 0.08750, 0.07448, 0.06985, 0.07398, 0.06032, 0.05496]
total_ce = [2.84456, 2.40543, 2.27699, 2.21640, 2.17922, 2.18447, 2.15529, 2.12161, 2.19438, 2.19590]
within_ce = [2.82199, 2.35560, 2.23892, 2.18955, 2.15859, 2.16624, 2.14077, 2.10834, 2.18230, 2.18443]
across_ce = [2.86712, 2.45527, 2.31507, 2.24325, 2.19985, 2.20269, 2.16981, 2.13489, 2.20646, 2.20736]

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs, kmeans_loss_view1)
plt.plot(epochs, kmeans_loss_view2)
plt.legend(["kmeans_loss_view1", "kmeans_loss_view2"])
plt.title("Kmeans clustering loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.savefig('/home/nano01/a/tao88/train_curve_1.png')

plt.subplot(1, 2, 2)
plt.plot(epochs, total_ce)
plt.plot(epochs, within_ce)
plt.plot(epochs, across_ce)
plt.legend(["total_ce", "within_ce", "across_ce"])
plt.title("Cross entropy loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig('/home/nano01/a/tao88/train_curve_new.png')

