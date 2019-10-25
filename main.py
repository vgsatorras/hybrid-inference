# from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import kalman, gnn
import settings
from datasets import synthetic, nclt, lorenz
import test
import losses
from datasets.dataloader import DataLoader
from utils import generic_utils as g_utils


def train_hybrid(args, net, device, train_loader, optimizer, epoch):
    net.train()
    stepsxsample = 1.0 * train_loader.dataset.total_len() / (len(train_loader.dataset) + 1e-12)
    for batch_idx, (ts, position, meas, x0, P0, operators) in enumerate(train_loader):
        position, meas, x0 = position.to(device), meas.to(device), x0.to(device)
        operators = g_utils.operators2device(operators, device)
        optimizer.zero_grad()
        outputs = net([operators, meas], x0, args.K, ts=ts)
        mse = F.mse_loss(outputs[-1], position)
        loss = losses.mse_arr_loss(outputs, position)
        if meas.size(0) == 1:
            loss = loss * meas.size(1) / stepsxsample
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tMSE: {:.6f}'.format(
                epoch, batch_idx * len(meas), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), mse.item()))


def adjust_learning_rate(optimizer, lr, epoch):
    new_lr = lr * (0.5 ** (epoch // (args.epochs/5)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def main_synthetic_hybrid(args, sigma=0.15, lamb=0.5, val_on_train=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("working on device %s" % device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset_test = synthetic.SYNTHETIC(partition='test', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                         equations="acceleration", gnn_format=True)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    dataset_train = synthetic.SYNTHETIC(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                        equations="acceleration", gnn_format=True)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = synthetic.SYNTHETIC(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                      equations="acceleration", gnn_format=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    (A, H, Q, R) = synthetic.create_model_parameters_v(s2_x=sigma ** 2, s2_y=sigma ** 2, lambda2=lamb ** 2)


    net = gnn.GNN_Kalman(args, A, H, Q, R, settings.x0_v, 0 * np.eye(len(settings.x0_v)), nf=args.nf,
                             prior=args.prior,  learned=args.learned, init=args.init, gamma=args.gamma).to(device)
    #print(net)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    min_val = test.test_gnn_kalman(args, net, device, val_loader)
    best_test = test.test_gnn_kalman(args, net, device, test_loader, plots=False)

    if val_on_train:
        train_mse = test.test_gnn_kalman(args, net, device, train_loader)
        min_val = (min_val * dataset_val.total_len() + train_mse * dataset_train.total_len()) / (dataset_val.total_len() + dataset_train.total_len())


    for epoch in range(1, args.epochs + 1):
        #adjust_learning_rate(optimizer, args.lr, epoch)
        train_hybrid(args, net, device, train_loader, optimizer, epoch)

        val_mse = test.test_gnn_kalman(args, net, device, val_loader)
        test_mse = test.test_gnn_kalman(args, net, device, test_loader, plots=False)

        if val_on_train:
            train_mse = test.test_gnn_kalman(args, net, device, train_loader)
            val_mse = (val_mse * dataset_val.total_len() + train_mse * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

        if val_mse < min_val:
            min_val, best_test = val_mse, test_mse

    print("Test loss: %.4f" % (best_test))
    return min_val.item(), best_test.item()


def main_synhtetic_kalman(args, sigma=0.1, lamb=0.5, val_on_train=False, optimal=False):
    if optimal:
        x0_format = 'a'
    else:
        x0_format = 'v'
    if val_on_train:
        dataset_train = synthetic.SYNTHETIC(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples,
                                            test_tt=args.test_samples,
                                            equations="acceleration", gnn_format=True, x0_format=x0_format)
        train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False)

    dataset_val = synthetic.SYNTHETIC(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                      equations="acceleration", gnn_format=True, x0_format=x0_format)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    dataset_test = synthetic.SYNTHETIC(partition='test', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples,
                                       equations="acceleration", gnn_format=True, x0_format=x0_format)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)


    print("Testing for sigma: %.3f \t lambda %.3f" % (sigma, lamb))
    if optimal:
        (A, H, Q, R) = synthetic.create_model_parameters_a()
        ks_v = kalman.KalmanSmoother(A, H, Q, R, settings.x0_a, 0 * np.eye(len(settings.x0_a)))
    else:
        (A, H, Q, R) = synthetic.create_model_parameters_v(T=1, s2_x=sigma ** 2, s2_y=sigma ** 2, lambda2=lamb ** 2)
        ks_v = kalman.KalmanSmoother(A, H, Q, R, settings.x0_v, 0 * np.eye(len(settings.x0_v)))
    print('Testing Kalman Smoother A')
    val_loss = test.test_kalman_nclt(ks_v, val_loader, plots=False)
    test_loss = test.test_kalman_nclt(ks_v, test_loader, plots=False)

    if val_on_train:
        train_loss = test.test_kalman_nclt(ks_v, train_loader, plots=False)
        val_loss = (val_loss * dataset_val.total_len() + train_loss * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

    return val_loss, test_loss


def main_lorenz_hybrid(args, sigma=2, lamb=0.5, val_on_train=False, dt=0.05, K=1, plot_lorenz=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("working on device %s" % device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    dataset_test = lorenz.LORENZ(partition='test', max_len=5000, tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt)
    test_loader = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    dataset_train = lorenz.LORENZ(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)

    dataset_val = lorenz.LORENZ(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    net = gnn.Hybrid_lorenz(args, sigma=sigma, lamb=lamb, nf=args.nf, dt=dt, K=K, prior=args.prior, learned=args.learned, init=args.init, gamma=args.gamma).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    min_val = test.test_gnn_kalman(args, net, device, val_loader)
    best_test = test.test_gnn_kalman(args, net, device, test_loader, plots=False, plot_lorenz=plot_lorenz)
    for epoch in range(1, args.epochs + 1):
        #adjust_learning_rate(optimizer, args.lr, epoch)
        train_hybrid(args, net, device, train_loader, optimizer, epoch)

        if epoch % args.test_every == 0:
            val_mse = test.test_gnn_kalman(args, net, device, val_loader)
            test_mse = test.test_gnn_kalman(args, net, device, test_loader, plots=False, plot_lorenz=plot_lorenz)

            if val_on_train:
                train_mse = test.test_gnn_kalman(args, net, device, train_loader)
                val_mse = (val_mse * dataset_val.total_len() + train_mse * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

            if val_mse < min_val:
                min_val, best_test = val_mse, test_mse

    print("Test loss: %.4f" % (best_test))
    return min_val.item(), best_test.item()


def main_lorenz_kalman(args, sigma=2, lamb=0.5, K=1, dt=0.05, val_on_train=False, plots=False):
    if val_on_train:
        dataset_train = lorenz.LORENZ(partition='train', tr_tt=args.tr_samples, val_tt=args.val_samples,
                                      test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataset_test = lorenz.LORENZ(partition='test', max_len=5000, tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt)
    loader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False)

    dataset_val = lorenz.LORENZ(partition='val', tr_tt=args.tr_samples, val_tt=args.val_samples, test_tt=args.test_samples, gnn_format=True, sparse=True, sample_dt=dt)
    loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)

    print("Testing for sigma: %.3f \t lambda %.3f" % (sigma, lamb))
    ks_v = kalman.ExtendedKalman_lorenz(K=K, sigma=sigma, lamb=lamb, dt=dt)
    print('Testing Kalman Smoother A')
    val_loss = test.test_kalman_lorenz(args, ks_v, loader_val, plots=False)
    test_loss = test.test_kalman_lorenz(args, ks_v, loader_test, plots=plots)

    if val_on_train:
        train_loss = test.test_kalman_lorenz(args, ks_v, loader_train, plots=False)
        val_loss = (val_loss * dataset_val.total_len() + train_loss * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

    return val_loss, test_loss


def main_nclt_hybrid(args, sx=0.15, sy=0.15, lamb=0.5, val_on_train=False):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    args.device = device
    print("working on device %s" % device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    dataset_train = nclt.NCLT(date='2012-01-22', partition='train', ratio=args.nclt_ratio)
    train_loader = DataLoader(dataset_train, batch_size=args.test_batch_size, shuffle=False)

    dataset_val = nclt.NCLT(date='2012-01-22', partition='val', ratio=args.nclt_ratio)
    val_loader = DataLoader(dataset_val, batch_size=args.test_batch_size, shuffle=False)

    dataset_test = nclt.NCLT(date='2012-01-22', partition='test', ratio=1.0)
    test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    (A, H, Q, R) = synthetic.create_model_parameters_v(s2_x=sx ** 2,  s2_y=sy ** 2, lambda2=lamb ** 2)
    net = gnn.GNN_Kalman(args, A, H, Q, R, settings.x0_v, 0 * np.eye(len(settings.x0_v)), nf=args.nf,
                         prior=args.prior,  learned=args.learned, init=args.init).to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    min_val = 1e8
    best_test = 1e8
    for epoch in range(1, args.epochs + 1):
        #adjust_learning_rate(optimizer, args.lr, epoch)
        train_hybrid(args, net, device, train_loader, optimizer, epoch)

        val_mse = test.test_gnn_kalman(args, net, device, val_loader)

        test_mse = test.test_gnn_kalman(args, net, device, test_loader)

        if val_on_train:
            train_mse = test.test_gnn_kalman(args, net, device, train_loader)
            val_mse = (val_mse * dataset_val.total_len() + train_mse * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

        if val_mse < min_val:
            min_val, best_test = val_mse, test_mse

    print("Test loss: %.4f" % (best_test))
    return min_val.item(), best_test.item()


def main_nclt_kalman(args, sx=0.15, sy=0.15, lamb=0.5, val_on_train=False):
    if val_on_train:
        dataset_train = nclt.NCLT(date='2012-01-22', partition='train')
        loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)

    dataset_val = nclt.NCLT(date='2012-01-22', partition='val')
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    dataset_test = nclt.NCLT(date='2012-01-22', partition='test')
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    print("Testing for sigma: (%.3f, %.3f) \t lambda %.3f" % (sx, sy, lamb))
    (A, H, Q, R) = synthetic.create_model_parameters_v(T=1., s2_x=sx ** 2, s2_y=sy ** 2, lambda2=lamb ** 2)
    ks_v = kalman.KalmanSmoother(A, H, Q, R, settings.x0_v, 0 * np.eye(len(settings.x0_v)))
    print('Testing Kalman Smoother A')
    val_loss = test.test_kalman_nclt(ks_v, loader_val, plots=False)
    test_loss = test.test_kalman_nclt(ks_v, loader_test, plots=False)

    if val_on_train:
        train_loss = test.test_kalman_nclt(ks_v, loader_train, plots=False)
        val_loss = (val_loss * dataset_val.total_len() + train_loss * dataset_train.total_len())/(dataset_val.total_len() + dataset_train.total_len())

    return val_loss, test_loss


if __name__ == '__main__':
    args = settings.get_settings()

    # main_kalman()
    # main_gnn()
    # main_synthetic_hybrid()
    # main_nclt_test(args)
    # main_nclt_hybrid(args)
    main_lorenz_hybrid(args)


