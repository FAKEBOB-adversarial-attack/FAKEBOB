
import numpy as np
import os
from kaldiHelper import kaldiHelper
import shutil
import copy

class openSetIdentificationModel(object):

    def __init__(self, group_id, spk_id_list, spk_model_list, ubm, 
                threshold=0., root_dir=".", system_dir="../GMM-UBM-model"):

        self.group_id = group_id
        #self.spk_model_list = spk_model_list
        self.ubm = ubm
        self.threshold = threshold

        """ make sure spk_id is sorted
        """
        spk_id_list_sorted = copy.deepcopy(spk_id_list)
        spk_model_list_sorted = copy.deepcopy(spk_model_list)
        spk_id_list_sorted.sort()
        for i, spk_id in enumerate(spk_id_list_sorted):
            index = spk_id_list.index(spk_id)
            spk_model_list_sorted[i] = spk_model_list[index]
        self.spk_id_list = spk_id_list_sorted
        self.spk_model_list = spk_model_list_sorted
        self.model_path = self.spk_model_list + [self.ubm]

        self.root_dir = root_dir
        self.parent_dir = root_dir + "/" + self.group_id
        self.audio_dir = self.parent_dir + "/" + "audio"
        self.mfcc_dir = self.parent_dir + "/" + "mfcc"
        self.log_dir = self.parent_dir + "/" + "log"
        self.score_dir = self.parent_dir + "/" + "score"
        self.system_dir = system_dir
        for child_dir in [self.audio_dir, self.mfcc_dir, self.log_dir, self.score_dir]:
            if not os.path.exists(child_dir):
                os.makedirs(child_dir)

    def score(self, audios, fs=16000, bits_per_sample=16, debug=False, n_jobs=28, mfcc_conf=None):

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
            audio_list = audios

        for i, audio in enumerate(audio_list):
            if audio.dtype != np.int16:
                audio_list[i] = (audio * (2 ** (bits_per_sample - 1))).astype(np.int16)
        
        kaldi_helper = kaldiHelper(self.root_dir, self.audio_dir, self.mfcc_dir, 
                                    self.log_dir, self.score_dir, self.system_dir)
        score_array = kaldi_helper.score(self.model_path, audio_list, debug=debug, n_jobs=n_jobs, fs=fs, bits_per_sample=bits_per_sample, mfcc_conf=mfcc_conf)

        #print(score_array)

        final_score = score_array[:, :-1] - score_array[:, -1:] # (n_audos, n_spks)

        return final_score if final_score.shape[0] > 1 else final_score[0] # (n_audios, n_spks) or (n_spks, )
    
    def make_decisions(self, audios, n_jobs=28, debug=False, fs=16000, bits_per_sample=16, mfcc_conf=None):

        reject = -1

        score = self.score(audios, debug=debug, n_jobs=n_jobs, fs=fs, bits_per_sample=bits_per_sample, mfcc_conf=mfcc_conf)
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



