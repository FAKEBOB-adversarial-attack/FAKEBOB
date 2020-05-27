
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import numpy as np
import os
from gmm_ubm_kaldiHelper import gmm_ubm_kaldiHelper
import shutil
import copy

class gmm_SV(object):

    def __init__(self, spk_id, model, ubm, pre_model_dir="pre-models", threshold=0.0):

        self.pre_model_dir = os.path.abspath(pre_model_dir)

        self.spk_id = os.path.abspath(spk_id)
        if not os.path.exists(self.spk_id):
            os.makedirs(self.spk_id)

        self.audio_dir = os.path.abspath(self.spk_id + "/audio")
        self.mfcc_dir = os.path.abspath(self.spk_id + "/mfcc")
        self.log_dir = os.path.abspath(self.spk_id + "/log")
        self.score_dir = os.path.abspath(self.spk_id + "/score")

        self.threshold = threshold

        self.utt_id = model[1]
        self.identity_location = model[2]
        
        self.model_list = [ubm, self.identity_location] # add ubm

        self.kaldi_helper = gmm_ubm_kaldiHelper(pre_model_dir=self.pre_model_dir, audio_dir=self.audio_dir,
                                                mfcc_dir=self.mfcc_dir, log_dir=self.log_dir, score_dir=self.score_dir)

    def score(self, audios, fs=16000, bits_per_sample=16, debug=False, n_jobs=5):

        if os.path.exists(self.audio_dir):
            shutil.rmtree(self.audio_dir)
        if os.path.exists(self.mfcc_dir):
            shutil.rmtree(self.mfcc_dir)
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        if os.path.exists(self.score_dir):
            shutil.rmtree(self.score_dir)

        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
        if not os.path.exists(self.mfcc_dir):
            os.makedirs(self.mfcc_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.score_dir):
            os.makedirs(self.score_dir)

        if isinstance(audios, np.ndarray):
            if len(audios.shape) == 1 or (len(audios.shape) == 2 and (audios.shape[0] == 1 or audios.shape[1] == 1)):
                audio_list = []
                audio_list.append(audios)
            elif len(audios.shape) == 2:
                audio_list = [audios[:, i] for i in range(audios.shape[1])]
            else:
                pass

        else:
            # audio_list = audios
            audio_list = copy.deepcopy(audios) # avoid influencing

        for i, audio in enumerate(audio_list):
            if audio.dtype != np.int16:
                audio_list[i] = (audio * (2 ** (bits_per_sample - 1))).astype(np.int16)
        
        score_array = self.kaldi_helper.score(self.model_list, audio_list, fs=fs, n_jobs=n_jobs, debug=debug, bits_per_sample=bits_per_sample)

        final_score = score_array[:, 1] - score_array[:, 0] # (n_audos, )

        return final_score if final_score.shape[0] > 1 else final_score[0] # (n_audios, ) or scalar

    def make_decisions(self, audios, fs=16000, bits_per_sample=16, n_jobs=5, debug=False):

        accept = 1
        reject = -1

        score = self.score(audios, fs=fs, bits_per_sample=bits_per_sample, debug=debug, n_jobs=n_jobs)
        if isinstance(score, np.ndarray):
            decisions = [accept if score_value >= self.threshold else reject for score_value in score]
        else:
            decisions = accept if score >= self.threshold else reject
        
        return decisions, score


