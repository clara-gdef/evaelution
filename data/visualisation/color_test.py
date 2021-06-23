from cycler import cycler
import matplotlib.pyplot as plt
import ipdb
import matplotlib.lines as mlines
from matplotlib.pyplot import cm
import numpy as np

with ipdb.launch_ipdb_on_exception():
    n = 20

    x = np.random.rand(n, 2)
    y = np.random.rand(n, 2)
    exps = [0, 1, 2] * 7
    inds = np.arange(20)
    color = cm.rainbow(np.linspace(0, 1, n))
    letters = "azertyuiopqsdfghjklmwxcvbn"
    color_legends = {k: v for k, v in enumerate(letters[:20])}
    shape_per_exp = {0: "x",
                     1: "s",
                     2: 'v'}
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f"TSNE projection of VAE space", fontsize=14)

    ax = fig.add_subplot(211)
    for num, i in enumerate(range(n)):
        ax.scatter(x[num, 0], x[num, 1],
                   color=color[inds[num]], cmap=color, marker=shape_per_exp[exps[num]])
        ax.set_title(f"TSNE of {len(x)} jobs 1 of split")

    ax = fig.add_subplot(212)
    for num, i in enumerate(range(n)):
        ax.scatter(y[num, 0], y[num, 1],
                   color=color[inds[num]], cmap=color, marker=shape_per_exp[exps[num]])
        ax.set_title(f"TSNE of {len(x)} jobs 2 of split")

    handles = []
    for k, v in color_legends.items():
        leg = mlines.Line2D([], [], color=color[k], linestyle='None', marker='o',
                            markersize=10, label=v)
        handles.append(leg)
    for k, v in shape_per_exp.items():
        leg = mlines.Line2D([], [], color='black', marker=shape_per_exp[k], linestyle='None',
                            markersize=10, label=v)
        handles.append(leg)

    fig.legend(handles=handles)
    ipdb.set_trace()
    plt.show()
