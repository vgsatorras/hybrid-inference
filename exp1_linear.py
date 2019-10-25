import settings
import main
import numpy as np
import utils.directory_utils as d_utils
import time

quick = False
args = settings.get_settings()
args.exp_name = str(time.time())+'_linear'
d_utils.init_folders(args.exp_name)

args.batch_size = 1
args.gamma = 0.03
args.test_samples = 10*1000
args.init = 'meas_invariant'


print(args)
lr_base = float(args.lr)
sweep_samples = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 10000]) * 10
epochs_arr = [40, 40, 60, 50, 50, 50, 40, 50, 50, 40, 25]
lr_arr = [lr_base/10, lr_base/5, lr_base/2] + [lr_base] * (len(epochs_arr) - 5)

best_sigma = 0.4

if len(sweep_samples) != len(epochs_arr):
    raise Exception('Arrays "sweep_samples" and "epochs_arr" do not match.')

def baseline():
    print("\n######## \nBaseline: start\n########\n")
    args.prior = False
    args.learned = False
    args.epochs = 1
    _, test = main.main_synthetic_hybrid(args)
    print("\n######## \nBaseline: end\n########\n")
    return test

def kalman(data):
    best_val = 1e8
    best_sigma = 0.15, 0.15
    test_error = 1e8
    args.tr_samples = int(data/2)
    args.val_samples = data - args.tr_samples
    # Best: sigma --> 0.063, lamb --> 0.4833
    for sigma in np.linspace(0.325, 0.5, 8):
        print("Kalman Smoother sigma %.4f" % sigma)
        val, test = main.main_synhtetic_kalman(args, sigma, val_on_train=True, optimal=False)
        if val < best_val:
            best_val = val
            best_sigma = sigma
            test_error = test
    print("Kalman Smoother error: %.4f" % test_error)
    return best_sigma, test_error

def kalman_optimal():
    _, test = main.main_synhtetic_kalman(args, sigma=0.1, val_on_train=False, optimal=True)
    return test

def only_prior(data=1000):
    args.K = 100
    args.tr_samples = int(data/2)
    args.val_samples = data - args.tr_samples
    args.prior = True
    args.learned = False
    args.epochs = 0
    best_val = 1e8
    best_sigma = 0.15
    test_error = 1e8
    # Best sigma --> 0.05, lambda --> 0.9

    for sigma in np.linspace(0.325, 0.5, 8):
        for lamb in np.linspace(0.5, 0.5, 1):
            print("Only prior sigma %.4f" % sigma)
            val, test = main.main_synthetic_hybrid(args, sigma, lamb=lamb, val_on_train=True)
            if val < best_val:
                best_val = val
                best_sigma = sigma
                test_error = test

    #    raise ('Sigma is in the limit')
    print("Only Prior error: %.4f" % test_error)
    return best_sigma, test_error


def hybrid(sigma, data=1000, epochs=1):
    args.K = 100
    args.tr_samples = int(data*0.5)
    args.val_samples = int(data - args.tr_samples)
    args.prior = True
    args.learned = True
    args.epochs = epochs
    best_val = 1e8
    test_error = 1e8
    val, test = main.main_synthetic_hybrid(args, sigma)
    if val < best_val:
        best_val = val
        test_error = test

    return test_error


def only_learned(data=1000, epochs=1):
    args.K = 100
    args.tr_samples = int(data*0.5)
    args.val_samples = int(data - args.tr_samples)
    args.prior = False
    args.learned = True
    args.epochs = epochs
    best_val = 1e8
    test_error = 1e8
    val, test = main.main_synthetic_hybrid(args)
    if val < best_val:
        best_val = val
        test_error = test

    return test_error


def only_learned_gnn(data=1000):
    args.tr_samples = int(data/2)
    args.val_samples = data - args.tr_samples
    args.epochs = epochs
    best_val = 1e8
    test_error = 1e8
    val, test = main.main_synthetic_hybrid(args, GNN=True)
    if val < best_val:
        best_val = val
        test_error = test

    return test_error


if __name__ == '__main__':
    ## Baseline ##
    base = baseline()
    results = {'baseline': base, 'prior': [], 'learned': [], 'hybrid': [], 'sigma': [], 'lamb':[], 'sigma_kalman': [], 'n_samples': [], 'kalman':[], 'kalman_optimal':[]}

    #results = {'prior': [], 'learned': [], 'hybrid': [], 'sigma': [], 'lamb':[], 'sigma_kalman': [], 'n_samples': [], 'kalman':[], 'kalman_optimal':[]}
    for n_samples, epochs, lr in zip(sweep_samples, epochs_arr, lr_arr):
        args.lr = lr
        print("### Linear experiment n_samples: %d \t epochs: %d" % (n_samples, epochs))

        ## Kalman Smoother Optimal ##
        print("\n######## \nKalman Optimal: start\n########\n")
        test_error = kalman_optimal()
        results['kalman_optimal'].append(test_error)
        print("\n######## \nKalman Optimal: end\n########\n")

        ## Kalman Smoother ##
        print("\n######## \nKalman Smoother: start\n########\n")
        sigma, test_error = kalman(n_samples)
        results['kalman'].append(test_error)
        results['sigma_kalman'].append(sigma)
        print("\n######## \nKalman Smoother: end\n########\n")


        ## Only Prior ##
        print("\n######## \nOnly Prior: start\n########\n")
        best_sigma, test_error = only_prior(n_samples)
        results['prior'].append(test_error)
        results['sigma'].append(best_sigma)
        print("\n######## \nOnly Prior: end\n########\n")
        print(best_sigma)

        ## Only Learned ##
        print("\n######## \nOnly Learned: start\n########\n")
        results['learned'].append(only_learned(n_samples, epochs))
        print("\n######## \nOnly Learned: end\n########\n")


        ## Hybrid ##
        print("\n######## \nHybrid: start\n########\n")
        test_error = hybrid(best_sigma, data=n_samples, epochs=int(epochs))
        #test_error = hybrid(0.425, data=n_samples, epochs=int(epochs))
        results['hybrid'].append(test_error)
        print("\n######## \nHybrid: end\n########\n")

        # Meta-informations

        results['n_samples'].append(n_samples)

        print('\nResults %s \n' % str(results))

    d_utils.write_file('logs/%s/log.txt' % (args.exp_name), str(results))


