import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
import scqubits
from tqdm import tqdm
from IPython.display import clear_output
from functools import partial
import matplotlib
import plotly.graph_objects as go
from plotly.io import write_image





import plotly.graph_objects as go
import numpy as np

z1 = 10**np.array([
    [8.83,8.89,8.81,8.87,8.9,8.87],
    [8.89,8.94,8.85,8.94,8.96,8.92],
    [8.84,8.9,8.82,8.92,8.93,8.91],
    [8.79,8.85,8.79,8.9,8.94,8.92],
    [8.79,8.88,8.81,8.9,8.95,8.92],
    [8.8,8.82,8.78,8.91,8.94,8.92],
    [8.75,8.78,8.77,8.91,8.95,8.92],
    [8.8,8.8,8.77,8.91,8.95,8.94],
    [8.74,8.81,8.76,8.93,8.98,8.99],
    [8.89,8.99,8.92,9.1,9.13,9.11],
    [8.97,8.97,8.91,9.09,9.11,9.11],
    [9.04,9.08,9.05,9.25,9.28,9.27],
    [9,9.01,9,9.2,9.23,9.2],
    [8.99,8.99,8.98,9.18,9.2,9.19],
    [8.93,8.97,8.97,9.18,9.2,9.18]
])

z2 = z1 + 1
z3 = z1 - 1

fig = go.Figure(data=[
    go.Surface(z=z1,x=np.arange(6),y=np.arange(15),showscale=False,colorscale ='Blues'),
    go.Surface(z=z2,x=np.arange(6),y=np.arange(15), showscale=False, opacity=0.9,colorscale ='Greens'),
    go.Surface(z=z3,x=np.arange(6),y=np.arange(15), showscale=False, opacity=0.9,colorscale ='Greys')

])
fig.update_layout(
    scene=dict(
        xaxis=dict(title=r'$E_J/E_L$',),
        yaxis=dict(title=r'$E_J/E_C$'),
        zaxis=dict(title=r'$log_{10}(t)ms$'),),
    title='integer 01 Lifetime', 
    width=700,
    height=700
)
fig.show()

write_image(fig, 'output_file1.pdf', format='pdf')
