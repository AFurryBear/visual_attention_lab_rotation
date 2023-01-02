# add the path to the abcTau package
import sys
sys.path.append('./abcTau')
# import the package
import abcTau
import numpy as np

import mat73
from scipy import stats
import pandas as pd
import os

# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["OPENBLAS_NUM_THREADS"] = "2"
# os.environ["MKL_NUM_THREADS"] = "2"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
# os.environ["NUMEXPR_NUM_THREADS"] = "2"

## define function
def comp_psd(x):
    """Compute the power spectrum density (PSD) using a Hamming window and direct fft.
    Parameters
    -----------
    x1 : nd array
        time-series from binned data (numTrials * numBin).
    T : float
        duration of each trial/time-series.
    deltaT : float
        temporal resolution of data (or binSize of spike counts).


    Returns
    -------
    psd : 1d array
        average  power spectrum density (PSD) across all trials.
    """
    n_points = len(x[0])
    x_windowed = (x - x.mean(1)[:, None]) * np.hamming(n_points)
    PSD = np.mean(np.abs(np.fft.rfft(x_windowed)) ** 2, axis=0)[1:-1]

    return PSD


def sort_sig_input(sig, atten_cond):
    ''' This function 1) sorts data to same shape; 2) creates OUT based on out1 and out2
    ------
    sig: data['Full'][layer][atten_cond][epoch] (chs x trials x 1024 timepoints)
    atten_cond: 'RF' or 'OUT'

    Returns
    ------
    sig_in (chs*trials x 1024 timepoints)
    '''
    if len(sig.shape) == 2:
        sig = sig.reshape((1,sig.shape[0],sig.shape[1])) ## sig (channel, trial, timepoints)
    if atten_cond == 'OUT':
        index = np.random.choice(sig.shape[1],int(sig.shape[1]/2))
        sig = sig[:, index, :]
    avg_sig = np.average(sig, axis=1) # average across trials # (channels, timepoints)
    sig_in = sig.reshape((sig.shape[0]*sig.shape[1],sig.shape[2])) -np.repeat(avg_sig, sig.shape[1], axis=0) # average across channels
    #             print(sig_in.shape)
    return sig_in


sid = sys.argv[1]
mid = int(sys.argv[2])

atten_cond = sys.argv[3]
epoch = sys.argv[4]
layer = sys.argv[5]


# extract statistics from real data
#datadir = '/mnt/qb/levina/yxiong34/annalab/data'
datadir = '/Volumes/NONAME/data_for_Yirong'

## get data information from csv
df = pd.read_csv(os.path.join(datadir, 'Subject_Info_corrected.csv'), index_col=0)
i = df.loc[(df.SessionID == sid) & (df.MonkeyID == mid) & (df.Region == 'V4')].index.values[0]
monkey_id = int(df.iloc[i]['MonkeyID'])
monkey_name = df.iloc[i]['MonkeyName']
region = df.iloc[i]['Region']

## load data
filepath = os.path.join(datadir, 'M%d%s-LFP/%s_%s_LFP_%s_sorted.mat' % (monkey_id, region, monkey_name, region, sid))
data = mat73.loadmat(filepath)['LfpStruct']['Sorted']

## set parameters
T_dic = {'PreFirstDimDataBiZSc': 1000, 'StationaryDataBiZSc': 430}

fs = 1017.375
T = T_dic[epoch]
deltaT = T / fs  # temporal resolution of data.
binSize = T / fs  # bin-size for binning data and computing the autocorrelation.

disp = None  # put the disperssion parameter if computed with grid-search
maxTimeLag = None  # only used when using autocorrelation for summary statistics


# select summary statistics metric
summStat_metric = 'comp_psd'
ifNorm = True  # if normalize the autocorrelation or PSD

# select generative model and distance function
generativeModel = sys.argv[6] #oneTauOU_oscil
distFunc = 'logarithmic_distance'

# set fitting params
epsilon_0 = 500  # initial error threshold
min_samples = 100  # min samples from the posterior
steps = 60  # max number of iterations
minAccRate = 0.01  # minimum acceptance rate to stop the iterations
parallel = False  # if parallel processing
n_procs = 1  # number of processor for parallel processing (set to 1 if there is no parallel processing)

## sort data into OUT and RF
if atten_cond == 'OUT':
    sig1 = sort_sig_input(data['Full'][layer]['OUT1'][epoch], atten_cond)
    sig2 = sort_sig_input(data['Full'][layer]['OUT2'][epoch], atten_cond)
    sig_in = np.concatenate((sig1, sig2), axis=0)
else:
    sig = data['Full'][layer][atten_cond][epoch]
    sig_in = sort_sig_input(sig, atten_cond)

## calc stats
data_mean = np.average(np.average(sig_in[:, 0:int(np.ceil(T*fs/1000))], axis=1), axis=0)
data_var = np.average(np.var(sig_in[:, 0:int(np.ceil(T*fs/1000))], axis=1), axis=0)
numTrials = sig_in.shape[0]
data_sumStat = comp_psd(sig_in[:, 0:int(np.ceil(T*fs/1000))])

# Define the prior distribution
# for a uniform prior: stats.uniform(loc=x_min,scale=x_max-x_min)
t_min = 0.0  # first timescale
t_max = T
prior = {'oneTauOU_oscil':[stats.uniform(loc=t_min, scale=t_max - t_min), stats.norm(loc=15, scale=1), stats.uniform(loc=0.2, scale=0.5)],
         'oneTauOU':[stats.uniform(loc=t_min, scale=t_max - t_min)],
         'twoTauOU':[stats.uniform(loc=t_min, scale=t_max - t_min), stats.uniform(loc=t_min, scale=t_max - t_min), stats.uniform(loc=0.2, scale=0.5)]}
priorDist = prior[generativeModel]

# path for loading and saving data
datasave_path = os.path.join(datadir, '../abc_results_%s_M%d_%s_%s/' % (sid, mid, atten_cond, epoch))
os.makedirs(datasave_path, exist_ok=True)

# path and filename to save the intermediate results after running each step
inter_save_direc = os.path.join(datadir, '../abc_results_%s_M%d_%s_%s/' % (sid, mid, atten_cond, epoch))
inter_filename = 'abc_intermediate_results_psd'

# load real data and define filenameSave
filenameSave = '%s_%s_M%d_%s_%s' % (generativeModel, sid, mid, atten_cond, epoch)

# creating model object
class MyModel(abcTau.Model):
    # This method initializes the model object.
    def __init__(self):
        pass

    # draw samples from the prior.
    def draw_theta(self):
        theta = []
        for p in self.prior:
            theta.append(p.rvs())
        return theta

    # Choose the generative model (from generative_models)
    # Choose autocorrelation computation method (from basic_functions)
    def generate_data(self, theta):
        # generate synthetic data
        if disp == None:
            syn_data, numBinData = eval('abcTau.generative_models.' + generativeModel + \
                                        '(theta, deltaT, binSize, T, numTrials, data_mean, data_var)')
        else:
            syn_data, numBinData = eval('abcTau.generative_models.' + generativeModel +
                                        '(theta, deltaT, binSize, T, numTrials, data_mean, data_var, disp)')

        # compute the summary statistics
        syn_sumStat = abcTau.summary_stats.comp_sumStat(syn_data, summStat_metric, ifNorm, deltaT, binSize,
                                                        T, numBinData, maxTimeLag)
        return syn_sumStat

    # Computes the summary statistics
    def summary_stats(self, data):
        sum_stat = data
        return sum_stat

    # Choose the method for computing distance (from basic_functions)
    def distance_function(self, data, synth_data):
        if np.nansum(synth_data) <= 0:  # in case of all nans return large d to reject the sample
            d = 10 ** 4
        else:
            d = eval('abcTau.distance_functions.' + distFunc + '(data, synth_data)')
        return d


# fit with aABC algorithm for any generative model
abc_results, final_step = abcTau.fit.fit_withABC(MyModel, data_sumStat, priorDist, inter_save_direc, inter_filename,
                                                 datasave_path, filenameSave, epsilon_0, min_samples,
                                                 steps, minAccRate, parallel, n_procs, disp)