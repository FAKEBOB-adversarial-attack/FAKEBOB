
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import numpy as np
import os
from gmm_ubm_kaldiHelper import gmm_ubm_kaldiHelper
import shutil
import copy

class gmm_OSI(object):

    def __init__(self, group_id, model_list, ubm, pre_model_dir="pre-models", threshold=0.0):

        self.pre_model_dir = os.path.abspath(pre_model_dir)

        self.group_id = os.path.abspath(group_id)
        if not os.path.exists(self.group_id):
            os.makedirs(self.group_id)

        self.audio_dir = os.path.abspath(self.group_id + "/audio")
        self.mfcc_dir = os.path.abspath(self.group_id + "/mfcc")
        self.log_dir = os.path.abspath(self.group_id + "/log")
        self.score_dir = os.path.abspath(self.group_id + "/score")

        self.threshold = threshold

        self.n_speakers = len(model_list)
        self.spk_ids = []
        self.utt_ids = []
        self.identity_locations = []

        for model in model_list:

            spk_id = model[0]
            utt_id = model[1]
            identity_location = model[2]

            self.spk_ids.append(spk_id)
            self.utt_ids.append(utt_id)
            self.identity_locations.append(identity_location)
        
        self.model_list = ([ubm] + self.identity_locations) # add ubm

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

        final_score = score_array[:, 1:] - score_array[:, 0:1] # (n_audos, n_spks)

        return final_score if final_score.shape[0] > 1 else final_score[0] # (n_audios, n_spks) or (n_spks, )
    
    def make_decisions(self, audios, fs=16000, bits_per_sample=16, n_jobs=5, debug=False):

        reject = -1

        score = self.score(audios, fs=fs, bits_per_sample=bits_per_sample, debug=debug, n_jobs=n_jobs)
        if len(score.shape) == 1:
            score = score[np.newaxis, :]

        max_score = np.max(score, axis=1)
        max_index = np.argmax(score, axis=1)
        decisions = list(max_index)
        for i, value in enumerate(max_score):
            if value < self.threshold:
                decisions[i] = reject
        
        if score.shape[0] == 1:
            decisions = decisions[0]
            score = score.flatten()

        return decisions, score



