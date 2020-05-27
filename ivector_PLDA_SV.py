
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import numpy as np
import os
import subprocess
import shlex
import shutil
from scipy.io.wavfile import read, write
import time
from ivector_PLDA_kaldiHelper import ivector_PLDA_kaldiHelper
import copy

fs = 16000
bits_per_sample = 16

class iv_SV:

    def __init__(self, spk_id, model, pre_model_dir="pre-models", threshold=0.0):
        
        self.pre_model_dir = os.path.abspath(pre_model_dir)

        self.spk_id = os.path.abspath(spk_id)
        if not os.path.exists(self.spk_id):
            os.makedirs(self.spk_id)

        self.audio_dir = os.path.abspath(self.spk_id + "/audio")
        self.mfcc_dir = os.path.abspath(self.spk_id + "/mfcc")
        self.log_dir = os.path.abspath(self.spk_id + "/log")
        self.ivector_dir = os.path.abspath(self.spk_id + "/ivector")

        self.threshold = threshold

        self.utt_id = model[1]
        self.identity_location = model[2]
        self.z_norm_mean = model[3] 
        self.z_norm_std = model[4]

        self.train_ivector_scp = self.spk_id + "/ivector.scp"
        np.savetxt(self.train_ivector_scp, np.concatenate((np.array([self.utt_id])[:, np.newaxis], np.array([self.identity_location])[:, np.newaxis]), axis=1), fmt="%s")

        self.kaldi_helper = ivector_PLDA_kaldiHelper(pre_model_dir=self.pre_model_dir, audio_dir=self.audio_dir,
                                                mfcc_dir=self.mfcc_dir, log_dir=self.log_dir,
                                                ivector_dir=self.ivector_dir)

    def score(self, audio_list, fs=16000, bits_per_sample=16, n_jobs=10, debug=False):
        

        if os.path.exists(self.audio_dir):
            shutil.rmtree(self.audio_dir)
        if os.path.exists(self.mfcc_dir):
            shutil.rmtree(self.mfcc_dir)
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        if os.path.exists(self.ivector_dir):
            shutil.rmtree(self.ivector_dir)

        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        if not os.path.exists(self.mfcc_dir):
            os.makedirs(self.mfcc_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.ivector_dir):
            os.makedirs(self.ivector_dir)
        
        if isinstance(audio_list, np.ndarray):
            if len(audio_list.shape) == 1 or (len(audio_list.shape) == 2 and (audio_list.shape[0] == 1 or audio_list.shape[1] == 1)):
                audio_list = [audio_list]
            else:
                audio_list = [audio_list[:, i] for i in range(audio_list.shape[1])]
        
        else:
            audio_list = copy.deepcopy(audio_list) # avoid influencing

        for i, audio in enumerate(audio_list):
            if not audio.dtype == np.int16:
                audio_list[i] = (audio * (2 ** (bits_per_sample - 1))).astype(np.int16)

        score_array = self.kaldi_helper.score(audio_list, [self.utt_id], n_jobs=n_jobs, flag=1, train_ivector_scp=self.train_ivector_scp, debug=debug)

        score_array = (score_array - self.z_norm_mean) / self.z_norm_std

        return score_array if score_array.size > 1 else score_array[0] # (n_audios, ) or scalar

    def make_decisions(self, audio_list, fs=16000, bits_per_sample=16, n_jobs=10, debug=False):

        accept = 1
        reject = -1
        
        scores = self.score(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        if isinstance(scores, np.ndarray):
            decisions = [accept if score >= self.threshold else reject for score in scores]
        else:
            decisions = accept if scores >= self.threshold else reject

        return decisions, scores
    
    def make_decisions_value(self, score):

        accept = 1
        reject = -1
        
        if score < self.threshold:
            return reject
        else:
            return accept

        







