import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb

# device_ids = [4]

class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N, device_ids=[0]):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda(device=device_ids[0])
        two_pi_n_over_N = two_pi_n_over_N.cuda(device=device_ids[0])
        hanning = hanning.cuda(device=device_ids[0])#此处应该注释
        # device=device_ids[0]
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2
        # print(complex_absolute.shape)
        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None, device_ids=[0]):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N, device_ids)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
        
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs,device_ids):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda(device=device_ids[0])
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range, device_ids)
        # print(complex_absolute.shape)
        # print(target.shape)
        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)
        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs,device_ids):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda(device=device_ids[0])

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
               
    '''
    @staticmethod
    def complex_absolute_on_harmonics(output, Fs, cuda=False):
        output = output.view(1, -1)

        N = output.size()[1]
        k = Variable(torch.arange(0, N), requires_grad=True)
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N, cuda)

        complex_absolute = (1 / (Fs * N)) * complex_absolute[:, 0:int(np.floor(N / 2.0))]
        complex_absolute[:, 1:-2] = 2 * complex_absolute[:, 1:-2]

        return complex_absolute

    @staticmethod
    def likelihoods_in_hr_freq(output, Fs, cuda=False):
        output = output.view(1, -1)
        N = output.size()[1]
        points_per_hz = float(N) / Fs

        complex_absolute = TorchLossComputer.complex_absolute_on_harmonics(output, Fs, cuda)

        low_pass = 0.67

        low = int(round(low_pass * points_per_hz))
        high = int(min(round(4 * points_per_hz) + 1, complex_absolute.numel()))
        complex_absolute = complex_absolute[:, low:high]

        return F.softmax(complex_absolute.view(-1), dim=-1).view(-1), points_per_hz, int(
            round(low_pass * points_per_hz)) / points_per_hz

    @staticmethod
    def __compute_prediction_interval(likelihoods, lmb, cuda=True):
        low = 0
        high = 1
        shortest_dist = len(likelihoods)
        shortest_low = 0
        shortest_high = 1
        while high <= len(likelihoods):
            if high > low:
                if cuda:
                    sum_is_higher_than_lambda = (likelihoods[low:high].sum() > lmb).cpu().data.numpy()
                else:
                    sum_is_higher_than_lambda = likelihoods[low:high].sum() > lmb
                if sum_is_higher_than_lambda and high - low < shortest_dist:
                    shortest_dist = high - low
                    shortest_low = low
                    shortest_high = high
                low += 1
            else:
                high += 1

        return shortest_low, shortest_high

    @staticmethod
    def get_arg_whole_max_hr(signal, Fs, cuda=True):
        likelihoods, points_per_hz, likelihoods_beginning_f = TorchLossComputer.likelihoods_in_hr_freq(signal, Fs, cuda=cuda)

        whole_max_val, whole_max_idx = likelihoods.view(-1).max(0)
        f_max = likelihoods_beginning_f + (whole_max_idx.data.cpu().numpy() / points_per_hz)
        arg_whole_max_hr = 60.0 * f_max

        return Variable(torch.FloatTensor([arg_whole_max_hr]))

    @staticmethod
    def compute_prediction_interval(signal, Fs, lmb, plot=False, cuda=False):
        likelihoods, points_per_hz, likelihoods_beginning_f = TorchLossComputer.likelihoods_in_hr_freq(signal, Fs, cuda=cuda)
        if not cuda:
            likelihoods = likelihoods.data.numpy()

        return TorchLossComputer.compute_prediction_interval_from_likelihoods(likelihoods, points_per_hz, lmb, likelihoods_beginning_f,
                                                                              plot, cuda=cuda)

    @staticmethod
    def compute_prediction_interval_from_likelihoods(likelihoods, points_per_hz, lmb, likelihoods_beginning_f, plot=False, cuda=True):
        low, high = TorchLossComputer.__compute_prediction_interval(likelihoods, lmb, cuda)
        if cuda:
            max_val, max_idx = likelihoods[low:high].max(0)
            max_idx = max_idx.data.cpu().numpy()
        else:
            max_idx = likelihoods[low:high].argmax()

        f_max = likelihoods_beginning_f + ((low + max_idx) / points_per_hz)
        arg_max_hr = 60.0 * f_max

        if cuda:
            hr_range = Variable((torch.arange(low, high) / points_per_hz) + likelihoods_beginning_f).cuda()
            arg_max_hr = Variable(torch.FloatTensor([arg_max_hr])).cuda()
        else:
            hr_range = (np.arange(low, high) / points_per_hz) + likelihoods_beginning_f

        exp_hr = 60.0 * ((1.0 / likelihoods[low:high].sum()) * likelihoods[low:high] * hr_range).sum()

        if plot:
            return exp_hr, arg_max_hr, likelihoods, points_per_hz, low, high
        else:
            return exp_hr, arg_max_hr
    
    @staticmethod
    def hr_bpm(input, Fs, cuda=True):
        input = input.view(1, -1)
        bpm_range = torch.arange(40, 240)

        complex_absolute = TorchLossComputer.complex_absolute(input, Fs, bpm_range, cuda)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)

        return whole_max_idx + bpm_range[0]

    @staticmethod
    def mae(input, target, Fs, regularization_factor, cuda=True):
        input = input.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 240)

        hr_target = Variable((torch.median(target).data * 60.0) - bpm_range[0]).type(torch.LongTensor)
        complex_absolute = TorchLossComputer.complex_absolute(input, Fs, bpm_range, cuda)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)

        regularization_factor = (1.0 / regularization_factor)
        return torch.abs(Variable(torch.FloatTensor([regularization_factor * (hr_target.item() - whole_max_idx.item())])))
    '''
    
    '''
    @staticmethod
    def em_power_spectrum_loss(output, target, Fs, regularization_factor, min_loss_std, scale_factor, cuda=True, posteriors=None):
        output = output.view(1, -1)
        target = target.view(1, -1)

        hr_gt = Variable(torch.FloatTensor([torch.median(target).item() * 60.0]))
        N = output.size()[1]

        target_sigma = torch.var(target).item()
        target_mu = torch.mean(target).item()
        if target_sigma < min_loss_std ** 2:
            target_sigma = min_loss_std ** 2

        complex_absolute = TorchLossComputer.complex_absolute_on_harmonics(output, Fs, cuda=cuda)

        points_per_hz = float(N) / Fs
        target_mu *= points_per_hz
        target_sigma *= points_per_hz ** 2
        weights = (1 / (math.sqrt(2 * math.pi) * target_sigma)) * \
                  torch.exp(-((torch.arange(0, complex_absolute.numel()) - target_mu) ** 2 / (2.0 * target_sigma ** 2)))

        low = int(round(0.67 * points_per_hz))
        high = int(min(round(4 * points_per_hz) + 1, complex_absolute.numel()))
        complex_absolute = complex_absolute[:, low:high]

        likelihoods = F.softmax(complex_absolute, dim=-1).view(-1)
        lsm = F.log_softmax(complex_absolute, dim=-1).view(-1)

        exp_hr_9, arg_max_hr_9 = TorchLossComputer.compute_prediction_interval_from_likelihoods(likelihoods, points_per_hz, 0.9,
                                                                                                low / points_per_hz,
                                                                                                cuda=cuda)
        exp_hr_1, arg_max_hr_1 = TorchLossComputer.compute_prediction_interval_from_likelihoods(likelihoods, points_per_hz, 0.1,
                                                                                                low / points_per_hz,
                                                                                                cuda=cuda)

        whole_max_val, whole_max_idx = likelihoods.view(-1).max(0)
        arg_whole_max_hr = 60.0 * ((low + whole_max_idx.data.cpu().numpy()) / points_per_hz)
        arg_whole_max_hr = Variable(torch.FloatTensor([arg_whole_max_hr]))

        if cuda:
            weights = Variable(weights).cuda()
            hr_gt = hr_gt.cuda().view(1)
            arg_whole_max_hr = arg_whole_max_hr.cuda().view(1)

        score = - (weights[low:high].view(-1) * lsm).sum()
        if posteriors is not None:
            score = - (posteriors.view(-1) * lsm).sum()

        sq_error_hr = torch.cat(((hr_gt - arg_whole_max_hr) ** 2, (hr_gt - exp_hr_9) ** 2, (hr_gt - arg_max_hr_9.view(1)) ** 2,
                                 (hr_gt - exp_hr_1) ** 2, (hr_gt - arg_max_hr_1.view(1)) ** 2), dim=0).view(1, -1)

        return scale_factor * (complex_absolute.norm() - 0.75) ** 2 + score / regularization_factor, likelihoods, weights[
                                                                                                                  low:high] / weights[
                                                                                                                              low:high].sum(), sq_error_hr
    '''