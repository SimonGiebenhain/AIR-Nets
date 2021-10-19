from glob import glob
import pickle as pkl
import numpy as np
import argparse
import traceback
import os
import pandas as pd

repair = False

if __name__ == '__main__' and not repair:
    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )

    parser.add_argument('-generation_path', type=str)
    parser.add_argument('-voxel_input', dest='voxel_input', action='store_true')
    parser.add_argument('-pc_input', dest='voxel_input', action='store_false')
    parser.set_defaults(voxel_input=True)
    parser.add_argument('-res',type=int)
    parser.add_argument('-points',type=int)
    parser.add_argument('-data_type', required=True)
    parser.add_argument('-test_equiv', action='store_true')
    parser.set_defaults(rerun=False)

    args = parser.parse_args()

    if args.data_type == 'human':
        generation_paths = glob(args.generation_path + '/scans/*/')
    else:
        generation_paths = glob(args.generation_path + "/*/*/")

    if args.data_type == 'ifnet':
        data_path = 'shapenet/data/'
    elif args.data_type == 'onet':
        data_path = 'occupancy_networks/data/ShapeNet'
    elif args.data_type == 'human':
        data_path = 'MPI-FAUST/test/scans'
    else:
        raise ValueError('Unexpected data type: ' + str(args.data_type))

    if args.test_equiv:
        transforms = [None, 'trans', 'scale', 'rot', 'rot_grav']
    else:
        transforms = [None]

    for transform in transforms:
        if transform is not None:
            print('Starting to evaluate transformation equivariance forperties for transform type ' + transform)
        eval_all = {
            'path' : [],
            'reconst_completeness': [],
            'reconst_accuracy': [],
            'reconst_normals completeness': [],
            'reconst_normals accuracy': [],
            'reconst_normals': [],
            'reconst_completeness2': [],
            'reconst_accuracy2': [],
            'reconst_chamfer_l2': [],
            'reconst_chamfer_l1': [],
            'reconst_f_score_05': [],
            'reconst_f_score_10': [],
            'reconst_f_score_15': [],
            'reconst_f_score_20': [],
            'reconst_iou': [],
            #'input_completeness': [],
            #'input_accuracy': [],
            #'input_normals completeness': [],
            #'input_normals accuracy': [],
            #'input_normals': [],
            #'input_completeness2': [],
            #'input_accuracy2': [],
            #'input_chamfer_l2': [],
            #'input_iou': []
        }

        eval_all_avg = {}
        eval_classes = {}
        eval_class_avg = {}

        if args.data_type == 'ifnet' or args.data_type == 'human':
            if transform is not None:
                eval_file_name = '/' + transform + '_' + 'eval.pkl'
            else:
                eval_file_name = '/eval.pkl'
        elif args.data_type == 'onet':
            if transform is not None:
                eval_file_name = '/' + transform + '_' + 'evalOnet.pkl'
            else:
                eval_file_name = '/evalOnet.pkl'

        count = 0
        for path in generation_paths:
            if os.path.exists(path + eval_file_name):
                try:
                    norm_path = os.path.normpath(path)
                    folder = norm_path.split(os.sep)[-2]
                    file_name = norm_path.split(os.sep)[-1]

                    eval_reconst = pkl.load(open(path + eval_file_name,'rb'))

                    eval_all['path'].append(path)
                    count += 1

                    for key in eval_reconst:
                        eval_all['reconst_' + key].append(eval_reconst[key])


                except Exception as err:
                    # logger.exception('Path: >>>{}<<<'.format(data['path'][0]))
                    print('Error with {}: {}'.format(path, traceback.format_exc()))
            else:
                print(path + eval_file_name + ' does not exist!')


        print('Eval based on {} samples!'.format(count))

        for key in eval_all:
            print(key)
            if not key == 'path':
                data = np.array(eval_all[key])
                data = data[~np.isnan(data)]
                if len(data) > 0:
                    print(np.mean(data))
                    eval_all_avg[key+'_mean'] = np.mean(data)
                    eval_all_avg[key + '_median'] = np.median(data)



        print('Starting class-wise evaluation!')

        # Separate evaluation for each object class
        for i, p in enumerate(eval_all['path']):
            p_splits = p.split('/')
            class_id = p_splits[-3]
            if class_id in eval_classes:
                for key in eval_all:
                    if not key == 'path':
                        eval_classes[class_id][key].append((p_splits[-2], eval_all[key][i]))
            else:
                print('Identified new class: ' + class_id)
                eval_classes[class_id] = {}
                for key in eval_all:
                    print(key)
                    if not key == 'path':
                        eval_classes[class_id][key] = [(p_splits[-2], eval_all[key][i])]


        # Calculate classwise average evaluation scores
        for class_id in eval_classes:
            eval_class_avg[class_id] = {}
            for categ in eval_classes[class_id]:
                print(categ)
                data = np.array([x[1] for x in eval_classes[class_id][categ]])
                data = data[~np.isnan(data)]
                if len(data) > 0:
                    print('Class' + class_id + 'has avage score of {}'.format(np.mean(data)))
                    eval_class_avg[class_id][categ + '_mean'] = np.mean(data)
                    eval_class_avg[class_id][categ + '_median'] = np.median(data)


        eval_df = pd.DataFrame(eval_all_avg ,index=[0])

        if transform is not None:
            pkl.dump(eval_all, open(args.generation_path + '/../{}_evaluation_results_{}.pkl'.format(transform, args.data_type), 'wb'))
            pkl.dump(eval_all_avg, open(args.generation_path + '/../{}_evaluation_results_avg_{}.pkl'.format(transform, args.data_type), 'wb'))
            pkl.dump(eval_classes, open(args.generation_path + '/../{}_evaluation_results_classes_{}.pkl'.format(transform, args.data_type), 'wb'))
            pkl.dump(eval_class_avg, open(args.generation_path + '/../{}_evaluation_results_class_avg_{}.pkl'.format(transform, args.data_type), 'wb'))
            eval_df.to_csv( args.generation_path + '/../{}_evaluation_results_{}.csv'.format(transform, args.data_type))
        else:
            pkl.dump(eval_all, open(args.generation_path + '/../evaluation_results_{}.pkl'.format(args.data_type), 'wb'))
            pkl.dump(eval_all_avg, open(args.generation_path + '/../evaluation_results_avg_{}.pkl'.format(args.data_type), 'wb'))
            pkl.dump(eval_classes, open(args.generation_path + '/../evaluation_results_classes_{}.pkl'.format(args.data_type), 'wb'))
            pkl.dump(eval_class_avg, open(args.generation_path + '/../evaluation_results_class_avg_{}.pkl'.format(args.data_type), 'wb'))
            eval_df.to_csv( args.generation_path + '/../evaluation_results_{}.csv'.format(args.data_type))

def repair_nans(path):

    pkl_file = pkl.load(open(path))

    for key in pkl_file:

        arr = np.array(pkl_file[key])
        arr = arr[~np.isnan(arr)]
        pkl_file[key] = arr

    eval_avg = {}

    for key in pkl_file:
        eval_avg[key] = pkl_file[key].sum() / len(pkl_file[key])

    pkl.dump(pkl_file , open(os.path.dirname(path) + '/eval_repaired.pkl', 'wb'))
    pkl.dump(eval_avg , open(os.path.dirname(path) + '/eval_avg_repaired.pkl', 'wb'))

