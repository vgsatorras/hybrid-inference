from __future__ import print_function
import sys, os
sys.path.append('../')
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import settings
import time
dates = [];
dates.append('2012-01-08')
dates.append('2012-01-15')
dates.append('2012-01-22')
dates.append('2012-02-02')
dates.append('2012-02-04')
dates.append('2012-02-05')
dates.append('2012-02-12')
dates.append('2012-02-18')
dates.append('2012-02-19')
dates.append('2012-03-17')
dates.append('2012-03-25')
dates.append('2012-03-31')
dates.append('2012-04-29')
dates.append('2012-05-11')
dates.append('2012-05-26')
dates.append('2012-06-15')
dates.append('2012-08-04')
dates.append('2012-08-20')
dates.append('2012-09-28')
dates.append('2012-10-28')
dates.append('2012-11-04')
dates.append('2012-11-16')
dates.append('2012-11-17')
dates.append('2012-12-01')
dates.append('2013-01-10')
dates.append('2013-02-23')
dates.append('2013-04-05')
dates = ['2012-01-22']
path_gps = "data/nclt/sensor_data/%s/gps.csv"
path_gps_rtk = "data/nclt/sensor_data/%s/gps_rtk.csv"
path_gps_rtk_err = "data/nclt/sensor_data/%s/gps_rtk_err.csv"
path_gt = "data/nclt/ground_truth/groundtruth_%s.csv"
compact_path = "temp/nclt_%s.pickle"

class NCLT(data.Dataset):
    def __init__(self, date, partition='train', ratio=1.0):
        self.partition = partition
        self.ratio = ratio
        if not os.path.exists(compact_path % date):
            print("Loading NCLT dataset ...")
            self.gps, self.gps_rtk, self.gps_rtk_err, self.gt = self.__load_data(date)
            self.__process_data()
            self.dump(compact_path % date, [self.gps, self.gps_rtk, self.gps_rtk_err, self.gt])

        else:
            [self.gps, self.gps_rtk, self.gps_rtk_err, self.gt] = self.load(compact_path % date)

        if self.partition == 'train':
            indexes = [1, 3]
        elif self.partition == 'val':
            indexes = [0, 2]
        elif self.partition == 'test':
            indexes = [4, 5, 6]
        else:
            raise Exception('Wrong partition')


        self.gps = [self.gps[i].astype(np.float32) for i in indexes]
        self.gps_rtk = [self.gps_rtk[i].astype(np.float32) for i in indexes]
        self.gt = [self.gt[i].astype(np.float32) for i in indexes]

        self.cut_data()


        print("NCLT %s loaded: %d samples " % (partition, sum([x.shape[0] for x in self.gps_rtk])))

        self.operators_b = [self.__buildoperators_sparse(self.gps[i].shape[0]) for i in range(len(self.gps))]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (state, meas) where target is index of the target class.
        """
        x0, P0 = self.__pos2x0(self.gps_rtk[index][0, 1:].astype(np.float32))
        return self.gt[index][:, 0], self.gt[index][:, 1:], self.gps_rtk[index][:, 1:], x0, P0, self.operators_b[index]

    def cut_data(self):
        self.gps = [cut_array(e, self.ratio) for e in self.gps]
        self.gps_rtk = [cut_array(e, self.ratio) for e in self.gps_rtk]
        self.gt = [cut_array(e, self.ratio) for e in self.gt]

    def __pos2x0(self, pos):
        if settings.x0_v.shape[0] == 4:
            x0 = np.zeros(4).astype(np.float32)
            x0[0] = pos[0]
            x0[2] = pos[1]
            P0 = np.eye(4)*1
        else:
            x0 = np.zeros(6).astype(np.float32)
            x0[0] = pos[0]
            x0[3] = pos[1]
            P0 = np.eye(6)*1
        return x0, P0

    def dump(self, path, object):
        if not os.path.exists('temp'):
            os.makedirs('temp')
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            return pickle.load(f)

    def __len__(self):
        return len(self.gt)

    def total_len(self):
        total = 0
        for arr in self.gt:
            total += arr.shape[0]
        return total

    def _generate_sample(self, seed):
        np.random.seed(seed)

        if self.acceleration:
            return simulate_system(create_model_parameters_a, K=self.K, x0=self.x0)
        else:
            return simulate_system(create_model_parameters_v, K=self.K, x0=self.x0)

    def __buildoperators_sparse_old(self, nn=20):
        # Identity
        i = torch.LongTensor([[i, i] for i in range(nn)])
        v = torch.FloatTensor([1 for i in range(nn)])
        I = torch.sparse.FloatTensor(i.t(), v)

        #Message right
        i = torch.LongTensor([[i, i+1] for i in range(nn-1)] + [[nn-1, nn-1]])
        v = torch.FloatTensor([1 for i in range(nn-1)] + [0])
        mr = torch.sparse.FloatTensor(i.t(), v)

        #Message left
        i = torch.LongTensor([[0, nn-1]] + [[i+1, i] for i in range(nn-1)])
        v = torch.FloatTensor([0] + [1 for i in range(nn-1)])
        ml = torch.sparse.FloatTensor(i.t(), v)

        return [I, mr, ml]

    def __buildoperators_sparse(self, nn=20):
        # Message right to left
        m_left_r = []
        m_left_c = []

        m_right_r = []
        m_right_c = []

        m_up_r = []
        m_up_c = []

        for i in range(nn - 1):
            m_left_r.append(i)
            m_left_c.append((i + 1))

            m_right_r.append(i + 1)
            m_right_c.append((i))

        for i in range(nn):
            m_up_r.append(i)
            m_up_c.append(i + nn)

        m_left = [torch.LongTensor(m_left_r), torch.LongTensor(m_left_c)]
        m_right = [torch.LongTensor(m_right_r), torch.LongTensor(m_right_c)]
        m_up = [torch.LongTensor(m_up_r), torch.LongTensor(m_up_c)]

        return {"m_left": m_left, "m_right": m_right, "m_up": m_up}

    def __load_gps(self, path, date):
        df = pd.read_csv(path % date)
        df = df.iloc[:, [0, 3, 4]]
        return df.values

    def __load_gps_err(self, date):
        df = pd.read_csv(path_gps % date)
        df = df.iloc[:, 6]
        return df.values

    def __load_gt(self, date):
        df = pd.read_csv(path_gt % date)
        gt = df.iloc[:, [0, 2, 1]].values
        gt_err = df.iloc[:, [5, 4]].values
        return gt, gt_err

    def __load_gps_rtk_err(self, date):
        df = pd.read_csv(path_gps_rtk_err % date)
        return df.values

    def __compute_gps_err(self, gps, gt):
        return np.mean(np.square(gps - gt), axis=1)

    def __load_data(self, date):
        "We use the timestamp of gps_rtk which has the lowest frequency 1 Hz"
        gps = self.__load_gps(path_gps, date)
        gps_rtk = self.__load_gps(path_gps_rtk, date)
        gps_rtk_err = self.__load_gps_rtk_err(date)
        gt, _ = self.__load_gt(date)

        self.lat0 = gps_rtk[0, 1]
        self.lng0 = gps_rtk[0, 2]
        self.bias = [gt[0, 1], gt[0, 2]]

        gps_rtk_dec = self.__decompose(gps_rtk, date)
        gps_rtk_err_dec = self.__decompose(gps_rtk_err, date)

        gps_ar = []
        gt_ar = []
        gps_rtk_ar, gps_rtk_err_ar = [], []

        for gps_rtk_i, gps_rtk_err_i in zip(gps_rtk_dec, gps_rtk_err_dec):
            idxs = self.__filer_freq(gps_rtk_i[:, 0], f=1.)
            gps_rtk_ar.append(gps_rtk_i[idxs, :])
            gps_rtk_err_ar.append(gps_rtk_err_i[idxs, :])


            #Matching with GT
            idxs_gt = self.__match_tt(gps_rtk_ar[-1][:, 0], gt[:, 0])
            gt_ar.append(gt[idxs_gt, :])

            #Matching with gps
            idxs = self.__match_tt(gps_rtk_ar[-1][:, 0], gps[:, 0])
            gps_ar.append(gps[idxs, :])

        return gps_ar, gps_rtk_ar, gps_rtk_err_ar, gt_ar

    def __decompose(self, data, date):
        if date == '2012-01-22':
            return [data[100:2054], data[2054:4009], data[4147:6400], data[6400:8890], data[9103:10856], data[11113:12608],
                    data[12733:13525]]#, [0, 4147, 9103, 11113, 12733]
        else:
            return data

    def concatenate(self, arrays):
        return np.concatenate(arrays, axis=0)

    def __process_data(self):
        '''
        lat0 = self.gps_rtk[0][0, 1]
        lng0 = self.gps_rtk[0][0, 2]
        bias = [self.gt[0][0, 1], self.gt[0][0, 2]]
        '''

        for i in range(len(self.gps_rtk)):
            self.gps_rtk[i][:, 1:] = polar2cartesian(self.gps_rtk[i][:, 1], self.gps_rtk[i][:, 2], self.lat0,
                                                     self.lng0)
            self.gps[i][:, 1:] = polar2cartesian(self.gps[i][:, 1], self.gps[i][:, 2], self.lat0,
                                                 self.lng0)

            self.gt[i][:, 1:] = remove_bias(self.gt[i][:, 1:], self.bias)

    def __match_tt(self, tt1, tt2):
        print("\tMatching gps and gt timestamps")
        arr_idx = []
        for i, ti in enumerate(tt1):
            diff = np.abs(tt2 - ti)
            min_idx = np.argmin(diff)
            arr_idx.append(min_idx)
        return arr_idx

    def _match_gt_step1(self, gps, gps_err, gt, margin=5):
        gt_aux = gt.copy()
        min_err = 1e10
        min_x, min_y = 0, 0
        for x in np.linspace(-margin, margin, 200):
            for y in np.linspace(-margin, margin, 200):
                gt_aux[:, 0] = gt[:, 0] + x
                gt_aux[:, 1] = gt[:, 1] + y
                err = mse(gps, gps_err, gt_aux)
                if err < min_err:
                    min_err = err
                    min_x = x
                    min_y = y
                    #print("x: %.4f \t y:%.4f \t err:%.4f" % (min_x, min_y, err))

        print(err)
        print("Fixing GT bias x: %.4f \t y:%.4f \t error:%.4f" % (min_x, min_y, min_err))
        return (min_x, min_y)

    def _match_gt_step2(self, gt, err):
        (min_x, min_y) = err
        gt[:, 0] = gt[:, 0] + min_x
        gt[:, 1] = gt[:, 1] + min_y
        return gt

    def __filer_freq(self, ts, f=1., window=5):
        arr_idx = []
        last_id = 0
        arr_idx.append(last_id)
        check = False
        while last_id < len(ts) - window:
            rel_j = []
            for j in range(1, window):
                rel_j.append(np.abs(f - (ts[last_id+j] - ts[last_id])/1000000))
            last_id = last_id + 1 + np.argmin(rel_j)

            min_val = np.min(rel_j)
            if min_val > 0.05:
                check = True
            arr_idx.append(last_id)
        if check:
            print("\tWarning: Not all frequencies are %.3fHz" % f)
        print("\tFiltering finished!")
        return arr_idx


def mse(gps, gps_err, gt, th=2):
    error = np.mean(np.square(gps - gt), axis=1)
    mapping = (gps_err < th).astype(np.float32)
    return np.mean(error*mapping)

def polar2cartesian(lat, lng, lat0, lng0):
    dLat = lat - lat0
    dLng = lng - lng0

    r = 6400000  # approx. radius of earth (m)
    x = r * np.cos(lat0) * np.sin(dLng)
    y = r * np.sin(dLat)
    return np.concatenate((np.expand_dims(x, 1), np.expand_dims(y, 1)), 1)


def remove_bias(vector, bias):
    for i in range(vector.shape[1]):
        vector[:, i] = vector[:, i] - bias[i]
    return vector

if __name__ == '__main__':
    for date in dates:
        dataset = NCLT('2012-01-22', partition='train')
        dataset = NCLT('2012-01-22', partition='val')
        dataset = NCLT('2012-01-22', partition='test')


def cut_array(array, ratio):
    length = len(array)
    return array[0:int(round(ratio*length))]