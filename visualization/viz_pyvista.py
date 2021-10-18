import pyvista as pv
import numpy as np
import os


my_theme = pv.themes.DefaultTheme()
my_theme.background = 'white'
pv.global_theme.load_theme(my_theme)

#folder = '../recs/FOLDER_TO_RECS' #todo
folder = '../recs/trainIF'
files = os.listdir(folder)

for f in files:
    if f.startswith('.') or f == 'trainIF_human':
        continue
    for i, ff in enumerate(os.listdir(folder + '/' + f)):
        m = pv.read(folder + '/' + f + '/' + ff + '/surface_reconstruction.off')

        data = np.load('{}/{}/{}/data.npz'.format(folder, f, ff), allow_pickle=True)['arr_0'].item()
        input_pc = data['inputs'].squeeze().numpy()
        pc = pv.PolyData(input_pc)

        rot_axes = np.eye(4)
        rot_axes_c = rot_axes.copy()
        rot_axes[:, 2], rot_axes[:, 1] = rot_axes_c[:, 1], rot_axes_c[:, 2]
        m.transform(rot_axes)
        pc.transform(rot_axes)
        pl = pv.Plotter()
        pl.add_mesh(m)
        pl.add(input_pc)
        pl.show()
