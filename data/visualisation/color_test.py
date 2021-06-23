from cycler import cycler
import matplotlib.pyplot as plt
import ipdb
from matplotlib.pyplot import cm
import numpy as np

with ipdb.launch_ipdb_on_exception():
    n = 20

    x = np.arange(n)
    y = np.arange(n)

    color=iter(cm.rainbow(np.linspace(0,1,n)))
    for i in range(n):
       c=next(color)
       plt.scatter(x, y,c=c)
    ipdb.set_trace()