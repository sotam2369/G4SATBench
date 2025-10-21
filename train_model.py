import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse
import csv

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from g4satbench.utils.options import add_model_options
from g4satbench.utils.utils import set_seed, safe_log, safe_div
from g4satbench.utils.logger import Logger
from g4satbench.utils.format_print import FormatTable
from g4satbench.data.dataloader import get_dataloader
from g4satbench.models.gnn import GNN
from torch_scatter import scatter_sum
from g4satbench.utils.folder_manager import SGATFolder
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, choices=['satisfiability', 'assignment', 'core_variable'], help='Experiment task')
    parser.add_argument('train_dir', type=str, help='Directory with training data')
    parser.add_argument('--train_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Category of the training data')
    parser.add_argument('--train_sample_size', type=int, default=None, help='The number of instance in each training splits')
    parser.add_argument('--checkpoint', type=str, default=None, help='pretrained checkpoint')
    parser.add_argument('--valid_dir', type=str, default=None, help='Directory with validating data')
    parser.add_argument('--valid_splits', type=str, nargs='+', choices=['sat', 'unsat', 'augmented_sat', 'augmented_unsat'], default=None, help='Category of the validating data')
    parser.add_argument('--valid_sample_size', type=int, default=None, help='The number of instance in each validating splits')
    parser.add_argument('--label', type=str, choices=[None, 'satisfiability', 'assignment', 'core_variable'], default=None, help='Label')
    parser.add_argument('--data_fetching', type=str, choices=['parallel', 'sequential'], default='parallel', help='Fetch data in sequential order or in parallel')
    parser.add_argument('--loss', type=str, choices=[None, 'supervised', 'unsupervised_1', 'unsupervised_2'], default=None, help='Loss type for assignment prediction')
    parser.add_argument('--save_model_epochs', type=int, default=1, help='Number of epochs between two model savings')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-8, help='L2 regularization weight')
    parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')
    parser.add_argument('--lr_step_size', type=int, default=50, help='Learning rate step size')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--lr_patience', type=int, default=10, help='Learning rate patience')
    parser.add_argument('--clip_norm', type=float, default=1.0, help='Clipping norm')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--out_dir', type=str, default=None, help='Base output directory for runs (overrides runs/)')

    add_model_options(parser)

    opts = parser.parse_args()

    set_seed(opts.seed)

    difficulty, dataset = tuple(os.path.abspath(opts.train_dir).split(os.path.sep)[-3:-1])
    splits_name = '_'.join(opts.train_splits)

    if opts.task == 'assignment':
        exp_name = f'train_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}_label={opts.label}_loss={opts.loss}/' + \
            f'graph={opts.graph}_init_emb={opts.init_emb}_model={opts.model}_n_iterations={opts.n_iterations}_lr={opts.lr:.0e}_weight_decay={opts.weight_decay:.0e}_seed={opts.seed}'
    else:
        exp_name = f'train_task={opts.task}_difficulty={difficulty}_dataset={dataset}_splits={splits_name}/' + \
            f'graph={opts.graph}_init_emb={opts.init_emb}_model={opts.model}_n_iterations={opts.n_iterations}_lr={opts.lr:.0e}_weight_decay={opts.weight_decay:.0e}_seed={opts.seed}'
    
    if opts.out_dir is not None:
        base_root = opts.out_dir
    else:
        base_root = 'runs'

    # If a custom out_dir was provided, always root the experiment there. Only when
    # no out_dir is provided do we preserve the old behavior of deriving the base
    # from the checkpoint path when resuming.
    if opts.out_dir is None and opts.checkpoint is not None:
        base_log_dir = os.path.abspath(os.path.join(opts.checkpoint, '../../', exp_name))
    else:
        base_log_dir = base_root

    # Create a per-run folder train_1, train_2, ... inside the experiment folder
    run_idx = 1
    while True:
        cand = os.path.join(base_log_dir, f'train_{run_idx}')
        if not os.path.exists(cand):
            opts.log_dir = cand
            break
        run_idx += 1

    opts.checkpoint_dir = os.path.join(opts.log_dir, 'checkpoints')

    os.makedirs(opts.log_dir, exist_ok=True)
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    opts.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(opts)

    model = GNN(opts)
    model.to(opts.device)

    if opts.checkpoint is not None:
        print('Loading model checkpoint from %s..' % opts.checkpoint)
        if opts.device.type == 'cpu':
            checkpoint = torch.load(opts.checkpoint, map_location='cpu')
        else:
            checkpoint = torch.load(opts.checkpoint)

        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # Initialize SGATFolder to manage outputs (logs, plots, csvs, best/last models)
    out_dir_param = opts.out_dir if opts.out_dir is not None else None
    work_folder = SGATFolder(directory=out_dir_param or '../plots/', b_weights=[1.0])
    # Set opts.log_dir and checkpoint_dir to the folder_manager path
    opts.log_dir = str(work_folder.folder_path)
    opts.checkpoint_dir = os.path.join(opts.log_dir, 'checkpoints')
    os.makedirs(opts.checkpoint_dir, exist_ok=True)

    # Replace manual logger with folder-managed log file
    opts.log = os.path.join(opts.log_dir, 'log.txt')
    sys.stdout = Logger(opts.log, sys.stdout)
    sys.stderr = Logger(opts.log, sys.stderr)

    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
    train_loader = get_dataloader(opts.train_dir, opts.train_splits, opts.train_sample_size, opts, 'train')

    if opts.valid_dir is not None:
        valid_loader = get_dataloader(opts.valid_dir, opts.valid_splits, opts.valid_sample_size, opts, 'valid')
    else:
        valid_loader = None

    if opts.scheduler is not None:
        if opts.scheduler == 'ReduceLROnPlateau':
            assert opts.valid_dir is not None
            scheduler = ReduceLROnPlateau(optimizer, factor=opts.lr_factor, patience=opts.lr_patience)
        else:
            assert opts.scheduler == 'StepLR'
            scheduler = StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_factor)

    # for printing
    if opts.task == 'satisfiability' or opts.task == 'core_variable':
        format_table = FormatTable()

    best_loss = float('inf')
    # histories for plotting/csv
    train_loss_history = []
    valid_loss_history = []
    train_eval_history = []
    valid_eval_history = []
    # track sum of per-instance satisfied-clause ratios (for unweighted average across problems)
    train_ratio_sum = 0.0
    valid_ratio_sum = 0.0
    for epoch in range(opts.epochs):
        print('EPOCH #%d' % epoch)
        print('Training...')
        train_loss = 0
        # per-epoch ratio accumulators (sum of per-instance ratios)
        train_ratio_sum = 0.0
        valid_ratio_sum = 0.0
        train_sat_sum = 0.0
        train_cnt = 0  # number of fully satisfied instances (legacy)
        train_tot = 0

        if opts.task == 'satisfiability' or opts.task == 'core_variable':
            format_table.reset()

        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(opts.device)
            batch_size = data.num_graphs

            if opts.task == 'satisfiability':
                pred = model(data)
                label = data.y
                loss = F.binary_cross_entropy(pred, label)
                format_table.update(pred, label)

            elif opts.task == 'assignment':
                c_size = data.c_size.sum().item()
                c_batch = data.c_batch
                l_edge_index = data.l_edge_index
                c_edge_index = data.c_edge_index
                
                v_pred = model(data)

                if opts.loss == 'supervised':
                    label = data.y
                    loss = F.binary_cross_entropy(v_pred, label)

                elif opts.loss == 'unsupervised_1':
                    # calculate the loss in Eq. 4 and Eq. 5
                    l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                    s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
                    s_max_nom = l_pred[l_edge_index] * s_max_denom

                    c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
                    c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
                    c_pred = safe_div(c_nom, c_denom)

                    s_min_denom = (-c_pred / 0.1).exp()
                    s_min_nom = c_pred * s_min_denom
                    s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
                    s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

                    score = safe_div(s_nom, s_denom)
                    loss = (1 - score).mean()

                elif opts.loss == 'unsupervised_2':
                    # calculate the loss in Eq. 6
                    l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                    l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
                    c_loss = -safe_log(1 - l_pred_aggr.exp())
                    loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()
                    
                v_assign = (v_pred > 0.5).float()
                l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                # per-instance satisfied clause counts (0..num_clauses)
                sat_counts = scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size).float()
                train_sat_sum += sat_counts.sum().item()
                # accumulate per-instance satisfied-clause ratios (sat_counts / per-instance clause counts)
                try:
                    train_ratio_sum += (sat_counts / data.c_size.float()).sum().item()
                except Exception:
                    # fallback: if data.c_size isn't a tensor or shape mismatch, approximate by total
                    train_ratio_sum += sat_counts.sum().item() / (c_size if c_size > 0 else 1)
                sat_batch = (sat_counts == data.c_size).float()
                train_cnt += sat_batch.sum().item()
            
            else:
                assert opts.task == 'core_variable'
                v_pred = model(data)
                v_cls = v_pred > 0.5
                label = data.y
                loss = F.binary_cross_entropy(v_pred, label)

                format_table.update(v_pred, label)

            train_loss += loss.item() * batch_size
            train_tot += batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts.clip_norm)
            optimizer.step()

        train_loss /= train_tot
        print('Training LR: %f, Training loss: %f' % (optimizer.param_groups[0]['lr'], train_loss))

        if opts.task == 'satisfiability' or opts.task == 'core_variable':
            format_table.print_stats()
        else:
            assert opts.task == 'assignment'
            # average satisfied clauses per instance for training
            train_avg_sat = train_sat_sum / train_tot if train_tot > 0 else 0.0
            train_acc = train_cnt / train_tot if train_tot > 0 else 0.0
            # average satisfied-clauses ratio (unweighted average over problems)
            train_avg_sat_ratio = train_ratio_sum / train_tot if train_tot > 0 else 0.0
            print('Training accuracy: %f' % train_acc)
            print('Training avg_satisfied_clauses: %f' % train_avg_sat)
            print('Training avg_satisfied_clauses_ratio: %f' % train_avg_sat_ratio)

        # append histories
        train_loss_history.append(train_loss)
        # valid_loss may not be set yet; append placeholder to keep lengths
        # valid_loss appended later after validation completes

        # if epoch % opts.save_model_epochs == 0:
        #     torch.save({
        #         'state_dict': model.state_dict(),
        #         'epoch': epoch,
        #         'optimizer': optimizer.state_dict()},
        #         os.path.join(opts.checkpoint_dir, 'model_%d.pt' % epoch)
        #     )

        if opts.valid_dir is not None:
            print('Validating...')
            valid_loss = 0
            valid_sat_sum = 0.0
            valid_cnt = 0
            valid_tot = 0

            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                format_table.reset()

            model.eval()
            for data in valid_loader:
                data = data.to(opts.device)
                batch_size = data.num_graphs
                with torch.no_grad():
                    if opts.task == 'satisfiability':
                        pred = model(data)
                        label = data.y
                        loss = F.binary_cross_entropy(pred, label)
                        format_table.update(pred, label)
                    
                    elif opts.task == 'assignment':
                        c_size = data.c_size.sum().item()
                        c_batch = data.c_batch
                        l_edge_index = data.l_edge_index
                        c_edge_index = data.c_edge_index

                        v_pred = model(data)

                        if opts.loss == 'supervised':
                            label = data.y
                            loss = F.binary_cross_entropy(v_pred, label)
                        
                        elif opts.loss == 'unsupervised_1':
                            # calculate the loss in Eq. 4 and Eq. 5
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            s_max_denom = (l_pred[l_edge_index] / 0.1).exp()
                            s_max_nom = l_pred[l_edge_index] * s_max_denom

                            c_nom = scatter_sum(s_max_nom, c_edge_index, dim=0, dim_size=c_size)
                            c_denom = scatter_sum(s_max_denom, c_edge_index, dim=0, dim_size=c_size)
                            c_pred = safe_div(c_nom, c_denom)

                            s_min_denom = (-c_pred / 0.1).exp()
                            s_min_nom = c_pred * s_min_denom
                            s_nom = scatter_sum(s_min_nom, c_batch, dim=0, dim_size=batch_size)
                            s_denom = scatter_sum(s_min_denom, c_batch, dim=0, dim_size=batch_size)

                            score = safe_div(s_nom, s_denom)
                            loss = (1 - score).mean()

                        elif opts.loss == 'unsupervised_2':
                            # calculate the loss in Eq. 6
                            l_pred = torch.stack([v_pred, 1 - v_pred], dim=1).reshape(-1)
                            l_pred_aggr = scatter_sum(safe_log(1 - l_pred[l_edge_index]), c_edge_index, dim=0, dim_size=c_size)
                            c_loss = -safe_log(1 - l_pred_aggr.exp())
                            loss = scatter_sum(c_loss, c_batch, dim=0, dim_size=batch_size).mean()

                        v_assign = (v_pred > 0.5).float()
                        l_assign = torch.stack([v_assign, 1 - v_assign], dim=1).reshape(-1)
                        c_sat = torch.clamp(scatter_sum(l_assign[l_edge_index], c_edge_index, dim=0, dim_size=c_size), max=1)
                        sat_counts = scatter_sum(c_sat, c_batch, dim=0, dim_size=batch_size).float()
                        valid_sat_sum += sat_counts.sum().item()
                        # accumulate per-instance satisfied-clause ratios
                        try:
                            valid_ratio_sum += (sat_counts / data.c_size.float()).sum().item()
                        except Exception:
                            valid_ratio_sum += sat_counts.sum().item() / (c_size if c_size > 0 else 1)
                        sat_batch = (sat_counts == data.c_size).float()
                        valid_cnt += sat_batch.sum().item()
                    
                    else:
                        assert opts.task == 'core_variable'
                        v_pred = model(data)
                        v_cls = v_pred > 0.5
                        label = data.y
                        loss = F.binary_cross_entropy(v_pred, label)

                        format_table.update(v_pred, label)

                valid_loss += loss.item() * batch_size
                valid_tot += batch_size

            valid_loss /= valid_tot
            print('Validating loss: %f' % valid_loss)

            if opts.task == 'satisfiability' or opts.task == 'core_variable':
                format_table.print_stats()
            else:

                assert opts.task == 'assignment'
                valid_acc = valid_cnt / valid_tot if valid_tot > 0 else 0.0
                valid_avg_sat = valid_sat_sum / valid_tot if valid_tot > 0 else 0.0
                # average satisfied-clauses ratio (unweighted average over problems)
                valid_avg_sat_ratio = valid_ratio_sum / valid_tot if valid_tot > 0 else 0.0
                print('Validating accuracy: %f' % valid_acc)
                print('Validating avg_satisfied_clauses: %f' % valid_avg_sat)
                print('Validating avg_satisfied_clauses_ratio: %f' % valid_avg_sat_ratio)

            # append validation histories (use satisfied-clause ratio as eval for assignment)
            valid_loss_history.append(valid_loss)
            if opts.task == 'assignment':
                # compute epoch-level train ratio (unweighted average across problems)
                train_avg_sat_ratio = train_ratio_sum / train_tot if train_tot > 0 else 0.0
                train_eval_history.append(train_avg_sat_ratio)
                valid_eval_history.append(valid_avg_sat_ratio)
            else:
                train_eval_history.append(train_acc if 'train_acc' in locals() else None)
                valid_eval_history.append(valid_acc if 'valid_acc' in locals() else None)

            # Use folder_manager to potentially update best model and save plots/csvs
            try:
                # save best/last model using appropriate metric (satisfied-clause ratio for assignment)
                if opts.task == 'assignment':
                    metric_tensor = torch.tensor(valid_avg_sat_ratio)
                else:
                    metric_tensor = torch.tensor(valid_acc if 'valid_acc' in locals() else 0.0)
                work_folder.save_model(model, metric_tensor, epoch)
            except Exception:
                # fallback: save raw state dicts
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'optimizer': optimizer.state_dict()}, os.path.join(opts.checkpoint_dir, f'model_{epoch}.pt'))

        else:
            # no validation; still append placeholders
            valid_loss = None
            valid_loss_history.append(valid_loss)
            if opts.task == 'assignment':
                train_eval_history.append(train_avg_sat_ratio if 'train_avg_sat_ratio' in locals() else None)
            else:
                train_eval_history.append(train_acc if 'train_acc' in locals() else None)
            valid_eval_history.append(None)
        # Scheduler step
        if opts.valid_dir is not None:
            if opts.scheduler is not None:
                if opts.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(valid_loss if valid_loss is not None else 0.0)
                else:
                    scheduler.step()
        else:
            if opts.scheduler is not None:
                scheduler.step()

        # Now save plots and csvs using folder_manager
        try:
            import numpy as _np
            work_folder.save_plot([_np.array(train_loss_history), _np.array([v for v in valid_loss_history if v is not None])], ['Train', 'Test'], 'loss')
            work_folder.save_plot([_np.array(train_eval_history), _np.array([v for v in valid_eval_history if v is not None])], ['Train', 'Test'], 'eval', add_border=max)
            # save_csv expects lists of lists
            work_folder.save_csv([train_loss_history], [ [v if v is not None else None for v in valid_loss_history] ], 'loss')
            work_folder.save_csv([train_eval_history], [ [v if v is not None else None for v in valid_eval_history] ], 'eval')
        except Exception:
            pass


if __name__ == '__main__':
    main()
