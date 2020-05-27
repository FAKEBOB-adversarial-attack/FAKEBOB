
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


from ivector_PLDA_kaldiHelper import ivector_PLDA_kaldiHelper
import numpy as np
import os
import shutil
import time
import copy

bits_per_sample = 16

class iv_OSI:

    def __init__(self, group_id, model_list, pre_model_dir="pre-models", threshold=0.0):
        
        self.pre_model_dir = os.path.abspath(pre_model_dir)

        self.group_id = os.path.abspath(group_id)
        if not os.path.exists(self.group_id):
            os.makedirs(self.group_id)

        self.audio_dir = os.path.abspath(self.group_id + "/audio")
        self.mfcc_dir = os.path.abspath(self.group_id + "/mfcc")
        self.log_dir = os.path.abspath(self.group_id + "/log")
        self.ivector_dir = os.path.abspath(self.group_id + "/ivector")

        self.threshold = threshold

        self.n_speakers = len(model_list)
        self.spk_ids = []
        self.utt_ids = []
        self.identity_locations = []
        self.z_norm_means = np.zeros(self.n_speakers, dtype=np.float64)
        self.z_norm_stds = np.zeros(self.n_speakers, dtype=np.float64)

        for i, model in enumerate(model_list):

            spk_id = model[0]
            utt_id = model[1]
            identity_location = model[2]
            mean = model[3]
            std = model[4]

            self.spk_ids.append(spk_id)
            self.utt_ids.append(utt_id)
            self.identity_locations.append(identity_location)
            self.z_norm_means[i] = mean
            self.z_norm_stds[i] = std
        
        ''' make sure self.ids is in order, otherwise kaldi may oder them, which may leads to wrong results
        '''
        self.spk_ids, self.utt_ids, self.identity_locations, self.z_norm_means, self.z_norm_stds = \
            self.order(self.spk_ids, self.utt_ids, self.identity_locations, self.z_norm_means, self.z_norm_stds)

        self.train_ivector_scp = self.group_id + "/ivector.scp"
        np.savetxt(self.train_ivector_scp, np.concatenate((np.array(self.utt_ids)[:, np.newaxis], np.array(self.identity_locations)[:, np.newaxis]), axis=1), fmt="%s")

        self.kaldi_helper = ivector_PLDA_kaldiHelper(pre_model_dir=self.pre_model_dir, audio_dir=self.audio_dir,
                                                     mfcc_dir=self.mfcc_dir, log_dir=self.log_dir, ivector_dir=self.ivector_dir)

    def order(self, spk_ids, utt_ids, identity_locations, z_norm_means, z_norm_stds):

        spk_ids_sort = copy.deepcopy(spk_ids)
        utt_ids_sort = copy.deepcopy(utt_ids)
        identity_locations_sort = copy.deepcopy(identity_locations)
        z_norm_means_sort = copy.deepcopy(z_norm_means)
        z_norm_stds_sort = copy.deepcopy(z_norm_stds)

        spk_ids_sort.sort()
        for i, spk_id in enumerate(spk_ids_sort):

            index = np.argwhere(np.array(spk_ids) == spk_id).flatten()[0]
            utt_ids_sort[i] = utt_ids[index]
            identity_locations_sort[i] = identity_locations[index]
            z_norm_means_sort[i] = z_norm_means[index]
            z_norm_stds_sort[i] = z_norm_stds[index]
        
        return spk_ids_sort, utt_ids_sort, identity_locations_sort, z_norm_means_sort, z_norm_stds_sort

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

        score_array = self.kaldi_helper.score(audio_list, self.utt_ids, n_jobs=n_jobs, flag=1, train_ivector_scp=self.train_ivector_scp, debug=debug)

        score_array = (score_array - self.z_norm_means) / self.z_norm_stds

        return score_array # (n_audios, n_spks) or (n_spks, ) when only one audio

    def make_decisions(self, audios, fs=16000, bits_per_sample=16, n_jobs=10, debug=False):

        reject = -1

        score_array = self.score(audios, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        if len(score_array.shape) == 1:
            score_array = score_array[np.newaxis, :]

        max_score = np.max(score_array, axis=1)
        max_index = np.argmax(score_array, axis=1)
        decisions = max_index
        for i, score in enumerate(max_score):
            if score < self.threshold:
                decisions[i] = reject
        decisions = list(decisions)

        if len(decisions) == 1:
            decisions = decisions[0]
            score_array = score_array.flatten()

        return decisions, score_array
