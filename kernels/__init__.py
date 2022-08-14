
import torch
from .utils import compute_label_acc
from .superparameters_search import HSGradientInversion
from .gradient_inversion import BaseGradientInversion
from .MLCO import MLCOGradientInversion
from .label_reconstruction import llg_label_recon

def reconstruction(
        x_pseudo_list, 
        y_pseudo,
        target_gradient,
        model,
        dm,
        ds,
        args, 
        metric_dict=None, 
        bn_loss_layers=None):

    lr_start_end = (args.lr, 1e-3)
    regularization = set_regularization(args)

    if args.superparameters_search:
        gi = HSGradientInversion(
                target_gradient,
                model,
                dm,
                ds,
                args)
        
        x_pseudo, loss_track = gi.run(
                                lr_start_end, 
                                regularization, 
                                x_pseudo_list, 
                                y_pseudo,
                                args.epochs, 
                                metric_dict=metric_dict, 
                                bn_loss_layers=bn_loss_layers)

    elif args.MinCombine:
        gi = MLCOGradientInversion(
                target_gradient, model,
                dm, ds, args)
        x_pseudo, loss_track = gi.multigroup_run(
            lr_start_end, regularization,
            x_pseudo_list, y_pseudo,
            bn_loss_layers=bn_loss_layers
        )

    else:
        gi = BaseGradientInversion(
                target_gradient, model,
                dm, ds, args)
        
        new_x_pseudo_list, loss_track_list = gi.multigroup_run(
            lr_start_end, regularization, 
            x_pseudo_list, y_pseudo, 
            args.epochs, bn_loss_layers=bn_loss_layers)

        # choose the result
        result_losses = torch.ones(size=(args.n_seed,))
        for i, loss_track in enumerate(loss_track_list):
            result_losses[i] = loss_track['gradient loss']
        index_best = torch.argmin(result_losses)
        x_pseudo = new_x_pseudo_list[index_best]
        print(f'choose the best results {index_best}.')
        print_str = f'[end results] > '
        for key, value in loss_track_list[index_best].items():
            print_str += f'[{key}]: {value:.4f} '
        print(print_str)
    
    return x_pseudo, loss_track

def set_regularization(args):

    regularization = dict()
    if args.TV > 0:
        regularization['TV'] = args.TV
    if args.BN > 0:
        regularization['BN'] = args.BN
    if args.input_norm != 0:
        regularization['input_norm'] = args.input_norm
    
    return regularization

def init_x_pseudo(args):

    '''return x_fake.shape = (n_seed, batch_size, n_channel, W, H)'''

    if not args.superparameters_search:
        x_pseudo_list = torch.randn(args.n_seed, *args.input_size)
    else:
        x_pseudo_list = torch.randn(*args.input_size)

    print(f'initialize list of x_pseudo: {x_pseudo_list.shape}')
    
    return x_pseudo_list

def get_y_pseudo(args, metric_dict, target_gradient):

    if args.pseudo_label_init == 'known':
        label_pred = metric_dict['y_true']

    elif args.pseudo_label_init == 'from_grads':
        pass
    elif args.pseudo_label_init == 'llg':
        label_pred = llg_label_recon(target_gradient, args.batch_size)
    
    print('Target:', metric_dict['y_true'])
    print('Recon.:', label_pred)

    n_correct, acc = compute_label_acc(metric_dict['y_true'], label_pred)
    print(f' > label acc: {acc * 100:.2f}%')

    return label_pred.view(-1,)

