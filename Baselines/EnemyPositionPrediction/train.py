from __future__ import print_function
import json, pickle
import argparse
import torch, sys
import os
sys.path.append("../..")
from Baselines.EnemyPositionPrediction.test import show_test_result, show_test_pro_result
from data_loader.BatchEnvEnemy import *
from Baselines.Net.ELPNet import ELPNet

from Baselines.EnemyPositionPrediction.train_spatial import TrainSpatial
from Baselines.EnemyPositionPrediction.train_global_spatial import TrainGlobalSpacial


ArgsELPNet = {'midplanes': 20,
                 'outplanes': 6,
                 'out_vector': 128,
                 'gnn_hidden': 3366,
                 'in_rnn': 512,
                 'out_rnn': 512}

torch.cuda.empty_cache()


def get_env(name):

    if name == 'ELPNet':
        env = Env_ELPNet()
    else:
        raise Exception('error')
    return env


def resume_training(model, args):
    model_latest_file = os.path.join(args.model_path, 'model_latest.pth')
    if os.path.isfile(model_latest_file):
        model.load_state_dict(torch.load(model_latest_file))
        print('Resume the training from {}'.format(model_latest_file))
    else:
        print('Resume missing, there are not files!')


def model_train(env, args):
    if args.net_name in ['ELPNet']:
        if args.module == 'full':
            trans_module = True
            gnn_module = True
        elif args.module == 'trans':
            trans_module = True
            gnn_module = False
        elif args.module == 'gnn':
            trans_module = False
            gnn_module = True
        elif args.module == 'base':
            trans_module = False
            gnn_module = False                
        
        model = ELPNet(env.n_channels, ArgsELPNet['midplanes'], ArgsELPNet['outplanes'], env.n_features,
                            ArgsELPNet['out_vector'], ArgsELPNet['in_rnn'], ArgsELPNet['out_rnn'],
                            env.frame_size[0] * env.frame_size[1], ArgsELPNet['gnn_hidden'], 
                            Trans_module=trans_module, GNN_module=gnn_module)
        if args.resume:
            resume_training(model, args)

        TrainGlobalSpacial.train(model, env, args)

    else:
        raise Exception('error')


def model_test(env, path, args):

    if args.net_name in ['ELPNet']:
        if args.module == 'full':
            trans_module = True
            gnn_module = True
        elif args.module == 'trans':
            trans_module = True
            gnn_module = False
        elif args.module == 'gnn':
            trans_module = False
            gnn_module = True
        elif args.module == 'base':
            trans_module = False
            gnn_module = False 
        
        model = ELPNet(env.n_channels, ArgsELPNet['midplanes'], ArgsELPNet['outplanes'], env.n_features,
                            ArgsELPNet['out_vector'], ArgsELPNet['in_rnn'], ArgsELPNet['out_rnn'],
                            env.frame_size[0] * env.frame_size[1], ArgsELPNet['gnn_hidden'], 
                            Trans_module=trans_module, GNN_module=gnn_module)
        model.load_state_dict(torch.load(path))
        if args.phrase == 'test':
            result = TrainGlobalSpacial.test(model, env, args)
        elif args.phrase == 'test_pro':
            result = TrainGlobalSpacial.test_pro(model, env, args)
        elif args.phrase == 'produe_feature_data':
            result = TrainGlobalSpacial.predect(model, env, args)

    else:
        raise Exception('error')
    return result


def next_path(model_folder, paths):
    models = {int(os.path.basename(model).split('.')[0].split('_')[-1])
              for model in os.listdir(model_folder) if 'latest' not in model}
    models_not_process = models - paths
    if len(models_not_process) == 0:
        return None
    models_not_process = sorted(models_not_process, reverse=True)
    paths.add(models_not_process[0])

    return os.path.join(model_folder, 'model_iter_{}.pth'.format(models_not_process[0]))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Enemy position prediction by the features of tensor: Wargame')
    parser.add_argument('--net_name', type=str, default='ELPNet',
                        help='the net is: ELPNet, add other networks for comparison')
    parser.add_argument('--piece_name', type=str, default='all',
                        help='the name of the chess pieces: tank1, tank2, car1, car2, soldier1, soldier2, all'),
    parser.add_argument('--replays_path', default='../../data/train_test/feature_tensor_vector',
                        help='Path for training, and test set')
    parser.add_argument('--race', default='red', help='Which race? (default: red)')
    parser.add_argument('--enemy_race', default='blue', help='Which the enemy race? (default: blue)')
    parser.add_argument('--phrase', type=str, default='train',
                        help='train|test_pro|test|produe_feature_data(default: train)')
    parser.add_argument('--gpu_id', default=0, type=int, help='Which GPU to use [-1 indicate CPU] (default: 0)')
    parser.add_argument('--module', type=str, default='full', 
                        help='full--Transformer and GNN, trans--Transformer module, gnn--GNN module, base--no module')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--n_steps', type=int, default=10, help='# of forward steps (default: 10)')
    parser.add_argument('--load_steps', type=int, default=1,
                        help='# the intervel steps of the frame of data (default: 1)')
    parser.add_argument('--n_replays', type=int, default=32, help='# of replays (default: 32)')
    parser.add_argument('--n_epochs', type=int, default=15, help='# of epoches (default: 10)')

    parser.add_argument('--save_intervel', type=int, default=50000,
                        help='Frequency of model saving (default: 50000)')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume the training (default: False)')
    args = parser.parse_args()

    args.name = args.net_name
    args.save_path = os.path.join('checkpoints', args.name, args.module, args.piece_name)
    print("args.save_path: ", args.save_path)
    args.model_path = os.path.join(args.save_path, 'snapshots')
    print("args.model_path: ", args.model_path)
    piece_name_to_offset = {'all': 0, 'tank1': 0, 'tank2': 1, 'car1': 2, 'car2': 3, 'soldier1': 4, 'soldier2': 5}

    args.piece_offset = piece_name_to_offset[args.piece_name]

    print('------------ Options -------------')
    for k, v in sorted(vars(args).items()):
        print('{}: {}'.format(k, v))
    print('-------------- End ----------------')

    
    if args.phrase == 'train':
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        
        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)
        
        with open(os.path.join(args.save_path, 'config'), 'w') as f:
            f.write(json.dumps(vars(args)))
        
        env = get_env(args.net_name)

        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
        # print("path_replays", path_replays)
        root = os.path.join('../')

        env.init(path_replays, root, args)

        model_train(env, args)  
        print('train ending')
    elif args.phrase == 'test':
        test_result_path = os.path.join(args.save_path, args.phrase)
        print(test_result_path)
        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)

        paths = set()
        test_result = []
        path_list = []
        while True:
            path = next_path(args.model_path, paths)
            if path is not None:
                print('[{}]Testing {} ...'.format(len(paths), path))
                env = get_env(args.net_name)
                path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format(args.phrase)))
                root = os.path.join('../')
                args.n_epochs = 1
                args.n_replays = 1
                env.init(path_replays, root, args)

                action_pre_per_replay, action_gt_per_replay = model_test(env, path, args)
                mean_acc, n_stage_acc = [], []
                for piece in range(6):
                    result = (action_pre_per_replay[piece], action_gt_per_replay[piece])
                    mean_acc_piece, n_stage_acc_piece = show_test_result(args.name, args.phrase, result,
                                                                         title=str(len(paths) - 1) + '-' + str(piece))
                    mean_acc.append(mean_acc_piece)
                    n_stage_acc.append(n_stage_acc_piece)
                dic = {'action_pre_per_replay': action_pre_per_replay,
                       'action_gt_per_replay': action_gt_per_replay,
                       'mean_acc': mean_acc,
                       'n_stage_acc': n_stage_acc}
                test_result.append(dic)
                path_list.append(path)
            else:
                with open(os.path.join(test_result_path, 'test_result'), 'wb') as f:
                    f.write(pickle.dumps(test_result))
                with open(os.path.join(test_result_path, 'path_list'), 'wb') as f:
                    f.write(pickle.dumps(path_list))
                print('Test ending')
                break
    elif args.phrase == 'test_pro':
        test_result_path = os.path.join(args.save_path, args.phrase)
        if not os.path.isdir(test_result_path):
            os.makedirs(test_result_path)

        if args.net_name == 'ELPNet':
            number = 800256
        else:
            raise Exception('error')

        path = os.path.join(args.model_path, 'model_iter_{}.pth'.format(number))

        print('Testing {} ...'.format(path))
        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format('test')))
        root = os.path.join('../')
        args.n_epochs = 1
        args.n_replays = 1
        env.init(path_replays, root, args)

        action_pre_per_replay, action_gt_per_replay = model_test(env, path, args)

        dic1 = {'action_pre_per_replay': action_pre_per_replay,
                'action_gt_per_replay': action_gt_per_replay}
        # 'mean_acc': mean_acc,
        # 'n_stage_acc': n_stage_acc}

        with open(os.path.join(test_result_path, 'test_result1'), 'wb') as f:
            f.write(pickle.dumps(dic1))

        mean_acc, n_stage_acc = [], []
        for piece in range(6):
            result = (action_pre_per_replay[piece], action_gt_per_replay[piece])
            mean_acc_piece, n_stage_acc_piece = show_test_pro_result(args.name, args.phrase, result)
            mean_acc.append(mean_acc_piece)
            n_stage_acc.append(n_stage_acc_piece)

        dic = {'action_pre_per_replay': action_pre_per_replay,
               'action_gt_per_replay': action_gt_per_replay,
               'mean_acc': mean_acc,
               'n_stage_acc': n_stage_acc}

        with open(os.path.join(test_result_path, 'test_result'), 'wb') as f:
            f.write(pickle.dumps(dic))
        print('Test_pro ending')

    elif args.phrase == 'produe_feature_data':
        save_data_path = os.path.join('../../data/predected_feature_data', args.net_name)
        if not os.path.isdir(save_data_path):
            os.makedirs(save_data_path)
        args.save_data_path = save_data_path

        if args.net_name == 'ELPNet':
            number = 1800519
        else:
            raise Exception('error')
        
        path = os.path.join(args.model_path, 'model_iter_{}.pth'.format(number))

        print('Testing data')
        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format('test')))
        root = os.path.join('../')
        args.n_epochs = 1
        args.n_replays = 1
        env.init(path_replays, root, args)

        result = model_test(env, path, args)

        print('Training data')
        env = get_env(args.net_name)
        path_replays = os.path.join(os.path.join(args.replays_path, '{}.json'.format('train')))
        root = os.path.join('../')
        args.n_epochs = 1
        args.n_replays = 1
        env.init(path_replays, root, args)

        result = model_test(env, path, args)


if __name__ == '__main__':
    main()
