import trimesh
import os


folder = '../recs/FOLDER_TO_RECS'

files = os.listdir(folder)


for f in files:
    if f.startswith('.'):
        continue
    for i, ff in enumerate(os.listdir(folder + '/' + f)):
            m = trimesh.load(folder + '/' + f + '/' + ff + '/surface_reconstruction.off')

            m.show()
