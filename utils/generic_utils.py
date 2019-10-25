import matplotlib.pyplot as plt
import math

def plot_trajectory(state, meas):
    if len(state[0]) == 4:
        x, y = 0, 2
    elif len(state[0]) == 6:
        x, y = 0, 3
    else:
        raise Exception('Wrong state')
    plt.figure(figsize=(7, 5))
    plt.plot(state[:, x], state[:, y], '-bo')
    plt.plot(meas[:, 0], meas[:, 1], 'rx')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['true state', 'observed measurement'])

    plt.axis('square')
    plt.tight_layout(pad=0)
    plt.show()

def plot_prediction(state, meas, est_state, est_cov):
    state_pos = state2position(state)
    est_state_pos = state2position(est_state)
    plt.figure(figsize=(7, 5))
    diag_var = cov2diag(est_cov)
    area_cov = [math.sqrt(diag_var[i][0])*math.sqrt(diag_var[i][1])*6000 for i in range(len(diag_var))]
    plt.scatter(est_state_pos[:, 0], est_state_pos[:, 1], s=area_cov, c=['#0000001F']*8)
    plt.plot(state[:, 0], state_pos[:, 1], '-bo')

    plt.plot(est_state_pos[:, 0], est_state_pos[:, 1], '-ko')
    plt.plot(meas[:, 0], meas[:, 1], ':rx')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['true state', 'inferred state', 'observed measurement'])

    plt.axis('square')
    plt.tight_layout(pad=0)
    plt.plot()
    plt.show()

def get_pos_index(state):
    if len(state[0]) == 4:
        return 0, 2
    elif len(state[0]) == 6:
        return 0, 3
    else:
        raise Exception('Error!')

def dim2indexpos(num_dimensions):
    if num_dimensions == 4:
        return 0, 2
    elif num_dimensions == 6:
        return 0, 3
    else:
        raise Exception('Error!')

def state2position(state):
    if len(state[0]) == 4:
        position = state[:, [0, 2]]
    elif len(state[0]) == 6:
        position = state[:, [0, 3]]
    else:
        raise Exception('Error!')
    return position


def cov2diag(cov):
    if len(cov[0, 0]) == 4:
        x, y = 0, 2
        #print(' It is 4')
    elif len(cov[0, 0]) == 6:
        x, y = 0, 3
        #print(' it is 6')
    else:
        raise Exception('Error!')
    diag_var = [(cov[i, x, x], cov[i, y, y])  for i in range(cov.shape[1])]
    return diag_var

def operators2device(operators, device):
    return [op.to(device) for op in operators]

def operators2device(operators, device):
    for key in operators:
        operators[key] = [op.to(device) for op in operators[key]]
    return operators
