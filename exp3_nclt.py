import settings
import main
import numpy as np
import utils.directory_utils as d_utils
import time

args = settings.get_settings()
args.exp_name = str(time.time())+'_nclt'
args.batch_size = 1
args.gamma = 0.03
args.init = 'meas_invariant'
epochs = 200
d_utils.init_folders(args.exp_name)

def baseline():
    args.prior = False
    args.learned = False
    args.epochs = 1
    _, test = main.main_nclt_hybrid(args)
    return test

def kalman():
    best_val = 1e8
    best_sigma = 0.15, 0.15
    best_lamb = 0.5
    test_error = 1e8
    for sx in np.linspace(0.05, 0.15, 10):
        for lamb in np.linspace(0.05, 0.81, 10):
            val, test = main.main_nclt_kalman(args, sx, sx, lamb, val_on_train=True)
            if val < best_val:
                best_val = val
                best_sigma = (sx, sx)
                best_lamb = lamb
                test_error = test
    return best_sigma, best_lamb, test_error


def only_prior(data=1000):
    args.K = 100
    args.tr_samples = 0
    args.val_samples = data
    args.prior = True
    args.learned = False
    args.epochs = 1
    best_val = 1e8
    best_sigma = 0.15
    best_lamb = 0.5
    test_error = 1e8
    for sx in np.linspace(0.16, 0.25, 10):
        for lamb in np.linspace(1.1, 3, 5):
            val, test = main.main_nclt_hybrid(args, sx, sx, lamb, val_on_train=True)
            if val < best_val:
                best_val = val
                best_sigma = (sx, sx)
                best_lamb = lamb
                test_error = test
    #if best_sigma == 0.7:
        #    raise ('Sigma is in the limit')
    return best_sigma, best_lamb, test_error


def only_learned():
    args.K = 100
    args.prior = False
    args.learned = True
    args.epochs = epochs
    _, test = main.main_nclt_hybrid(args)
    return test


def hybrid(sx=0.05, sy=0.05, lamb=0.9):
    args.K = 100
    args.prior = True
    args.learned = True
    args.epochs = epochs
    _, test = main.main_nclt_hybrid(args, sx, sy, lamb)
    return test


if __name__ == '__main__':

    results = {'baseline': [], 'prior': [], 'learned': [], 'hybrid': [], 'kalman': [], 'sigma_k': [], 'lamb_k': [], 'sigma': [], 'lamb': [], 'ratio': []}

    for ratio in np.linspace(1, 1., 1):
        args.nclt_ratio = ratio

        results['ratio'].append(ratio)
        results['baseline'] = baseline()

        ## kalman ##
        sigma, lamb, test_error = kalman()
        results['lamb_k'].append(lamb)
        results['sigma_k'].append(sigma)
        results['kalman'].append(test_error)


        ## Only Prior ##
        sigma, lamb, test_error = only_prior()
        results['lamb'].append(lamb)
        results['sigma'].append(sigma)
        results['prior'].append(test_error)

        ## Only Learned ##
        results['learned'].append(only_learned())

        ## Hybrid ##
        sx, sy = sigma
        results['hybrid'].append(hybrid(sx, sy, lamb))

        print(results)

