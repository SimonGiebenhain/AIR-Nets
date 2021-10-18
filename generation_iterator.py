import os
import traceback
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


# def gen_iterator(out_path, dataset, gen_p , buff_p, start,end):
def gen_iterator(out_path, dataset, gen_p, shuffle=True):

    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)

    # can be run on multiple machines if dataset is shuffled (already generated objects are skipped)
    loader = dataset.get_loader(shuffle=shuffle)

    data_tupels = []
    for i, data in tqdm(enumerate(loader)):
        path = os.path.normpath(data['path'][0])
        path_end = path.split(os.sep)[-1]
        splits = path_end.split('.')
        path_end = splits[0]

        export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path_end)

        # check if existing file is empty
        if os.path.exists(export_path + 'surface_reconstruction.off'):
            file = open(export_path + 'surface_reconstruction.off', "r")
            line_count = 0
            for line in file:
                line_count += 1
                if line_count > 5:
                    break
            if line_count > 5:
                print('Path exists - skip! {}'.format(export_path))
                continue
            if line_count < 5:
                print('Empty mesh detected! {}'.format(export_path))

        try:
            if len(data_tupels) >= 20:
                create_meshes(data_tupels)
                data_tupels = []
            logits = gen.generate_mesh(data)
            if 'input_pc' in data:
                data['inputs'] = data['input_pc']
                del data['input_pc']
            data_tupels.append((logits, data, export_path, ''))
        except Exception as err:
            print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))
    try:
        create_meshes(data_tupels)
    except Exception as err:
        print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))


def save_mesh(data_tupel):
    logits, data, export_path, name = data_tupel

    if gen.method == 'mcubes':
        mesh = gen.mesh_from_logits_mcubes(logits)
    else:
        mesh = gen.mesh_from_logits_MISE(logits)

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    exp_path = export_path + name + 'surface_reconstruction.off'
    mesh.export(exp_path)
    if not os.path.exists(export_path + 'data.npz'):
        np.savez(export_path + 'data.npz', data)


def create_meshes(data_tupels):
    p = Pool(10)
    p.map(save_mesh, data_tupels)
    p.close()
    p.join()