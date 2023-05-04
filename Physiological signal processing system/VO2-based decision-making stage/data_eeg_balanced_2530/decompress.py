import numpy as np

"""
Run to decompress the .npz files
"""

up = np.load('up_compressed.npz')['up']
down = np.load('down_compressed.npz')['down']
label = np.load('label_compressed.npz')['label']
Vr0 = np.load('Vr0_compressed.npz')['Vr0']
Vr1 = np.load('Vr1_compressed.npz')['Vr1']
Vr = np.concatenate([Vr0, Vr1])

np.save('up.npy', up)
np.save('down.npy', down)
np.save('label.npy', label)
np.save('Vr.npy', Vr)
