
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import numpy as np
import os
from scipy.io.wavfile import read, write
import subprocess
import shlex
import time

class ivector_PLDA_kaldiHelper:

    def __init__(self, pre_model_dir="pre-models", audio_dir=None, mfcc_dir=None, log_dir=None, ivector_dir=None):

        ''' pre_model_dir: directory where final.dubm, final.ubm, final.ie, 
                       mean.vec, transfrom.mat, plda, conf/, steps/, utils/ and sid/ are stored. 
                       We call all these pre models.
        audio_dir: directory to store the temp audios
        mfcc_dir: directory where generated mfcc is stored
        log_dir: directory where log information of kaldi is stored
        ivector_dir: directory where extracted ivectors are stored
        '''

        self.pre_model_dir = os.path.abspath(pre_model_dir)
        self.conf_dir = os.path.join(self.pre_model_dir, "conf")
        audio_dir = audio_dir if audio_dir else "audio"
        self.audio_dir = os.path.abspath(audio_dir)

        mfcc_dir = mfcc_dir if mfcc_dir else "mfcc"
        self.mfcc_dir = os.path.abspath(mfcc_dir)

        log_dir = log_dir if log_dir else "log"
        self.log_dir = os.path.abspath(log_dir)

        ivector_dir = ivector_dir if ivector_dir else "ivector"
        self.ivector_dir = os.path.abspath(ivector_dir)

        ''' deal with the protential permission issue
        '''
        all_files = (self.get_all_files(self.pre_model_dir + "/utils") + 
                     self.get_all_files(self.pre_model_dir + "/steps") + 
                     self.get_all_files(self.pre_model_dir + "/sid"))
            
        for file in all_files:
            change_permission_command = "chmod 777 " + file
            args = shlex.split(change_permission_command)
            p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.wait()
    
    def get_all_files(self, root_dir):

        files = []
        for name in os.listdir(root_dir):
            path = os.path.join(root_dir, name)
            if os.path.isfile(path):
                files.append(path)
            else:
                files_sub = self.get_all_files(path)
                files += files_sub
        
        return files

    def write_audio(self, audio_list, audio_dir=None, fs=16000):
        
        audio_dir = os.path.abspath(audio_dir) if audio_dir else self.audio_dir

        n_audios = len(audio_list)
        audio_path_list = list(range(n_audios))

        for i, audio in enumerate(audio_list):
            path = audio_dir + "/{}.wav".format(i)
            write(path, fs, audio)
            audio_path_list[i] = path
        
        return audio_path_list

    def data_prepare(self, audio_path_list, utt_id_list=None, spk_id_list=None, audio_dir=None, debug=False):

        ''' generate wav.scp, utt2spk, spk2utt in audio_dir according to utt_id_list,
            spk_id_list, audio_path_list. utt_id_list, spk_id_list, audio_path_list are
            three list objects with the same len
            Note: if utt_id_list and spk_id_listis provided, both of them must be in sorted order, 
            otherwise, the returning scores may disorder
        '''

        audio_dir = os.path.abspath(audio_dir) if audio_dir else self.audio_dir
        if not audio_dir:
            print("--- Error:audio_dir is None, quit data_prepare ---")
            return

        ''' if audio_path_list is in relative path, make it in absolute path, otherwise kaldi may cannot finds them.
        '''
        audio_path_list = [os.path.abspath(path) for path in audio_path_list]

        # assert len(utt_id_list) == len(spk_id_list)
        # assert len(utt_id_list) == len(audio_path_list)

        n_audios = len(audio_path_list)
        spk_id_list = spk_id_list if spk_id_list else [("0000" + str(i+1))[-5:] for i in range(n_audios)] # this may cause a bug if there are more than 100000 audios in audio_list
        utt_id_list = utt_id_list if utt_id_list else [spk_id + "-{}".format(1) for spk_id in spk_id_list]

        # wav.scp, utt2spk
        wav_file = audio_dir + "/wav.scp"
        utt2spk_file = audio_dir + "/utt2spk"
        wav_mat = np.concatenate((np.array(utt_id_list)[:, np.newaxis], np.array(audio_path_list)[:, np.newaxis]), axis=1)
        np.savetxt(wav_file, wav_mat, fmt="%s")
        # utt2spk_mat = wav_mat
        # utt2spk_mat[:, 1] = np.array(spk_id_list) # bug may rise here!
        utt2spk_mat = np.concatenate((np.array(utt_id_list)[:, np.newaxis], np.array(spk_id_list)[:, np.newaxis]), axis=1)
        np.savetxt(utt2spk_file, utt2spk_mat, fmt="%s")

        # spk2utt
        spk_id_vector = utt2spk_mat[:, 1]
        spk_id_unique_vector = np.unique(spk_id_vector)
        # may raise bug here, delete !!!
        # spk2utt_mat = np.concatenate((spk_id_unique_vector[:, np.newaxis], 
        #                               spk_id_unique_vector[:, np.newaxis]), axis=1)
        # for i, spk_id in enumerate(spk_id_unique_vector):
        #     utt_index = np.argwhere(spk_id_vector == spk_id).flatten()
        #     spk2utt_mat[i, 1] = ""
        #     for index in utt_index:
        #         spk2utt_mat[i, 1] += utt2spk_mat[index, 0]
        spk2utt_list = []
        for spk_id in spk_id_unique_vector:
            utt_index = np.argwhere(spk_id_vector == spk_id).flatten()
            utts = ""
            for index in utt_index:
                utts += (utt2spk_mat[index, 0] + " ")
            utts.rstrip()
            spk2utt_list.append(utts)

        spk2utt_mat = np.concatenate((spk_id_unique_vector[:, np.newaxis], np.array(spk2utt_list)[:, np.newaxis]), axis=1)
        spk2utt_file = audio_dir + "/spk2utt"
        np.savetxt(spk2utt_file, spk2utt_mat, fmt="%s")

        # fix dir
        ''' Note: every time we execuate the shell script in utils/, steps/ or sid/, 
            we need to change the current directory to pre_model_dir.
            To avoid mistask, we should make all the dirs (e.g., audio_dir, log_dir) in absoulute path
        '''
        current_dir = os.path.abspath(os.curdir)
        os.chdir(self.pre_model_dir)

        fix_dir_command = self.pre_model_dir + "/utils/fix_data_dir.sh " + audio_dir
        args = shlex.split(fix_dir_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        os.chdir(current_dir)

        return utt_id_list
    
    def make_mfcc(self, n_jobs=10, mfcc_conf=None, audio_dir=None, mfcc_dir=None, log_dir=None, debug=False):

        mfcc_conf = os.path.abspath(mfcc_conf) if mfcc_conf else (self.conf_dir + "/mfcc.conf")
        audio_dir = os.path.abspath(audio_dir) if audio_dir else self.audio_dir
        mfcc_dir = os.path.abspath(mfcc_dir) if mfcc_dir else self.mfcc_dir
        log_dir = os.path.abspath(log_dir) if log_dir else self.log_dir

        extract_mfcc_command = (self.pre_model_dir + "/steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config " +
                                mfcc_conf + " --nj " + str(n_jobs) + " --cmd '$train_cmd' " + audio_dir + " " + 
                                log_dir + " " + mfcc_dir)
        
        current_dir = os.path.abspath(os.curdir)
        os.chdir(self.pre_model_dir)

        args = shlex.split(extract_mfcc_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        os.chdir(current_dir)

    def compute_vad(self, n_jobs=10, vad_conf=None, audio_dir=None, vad_dir=None, log_dir=None, debug=False):

        vad_conf = os.path.abspath(vad_conf) if vad_conf else (self.conf_dir + "/vad.conf")
        audio_dir = os.path.abspath(audio_dir) if audio_dir else self.audio_dir
        vad_dir = os.path.abspath(vad_dir) if vad_dir else self.mfcc_dir
        log_dir = os.path.abspath(log_dir) if log_dir else self.log_dir

        vad_command = (self.pre_model_dir + "/steps/compute_vad_decision.sh --nj " + str(n_jobs) +
                        " --cmd '$train_cmd' --vad-config " + vad_conf + " " +
                        audio_dir + " " + log_dir + " " + vad_dir)
        
        current_dir = os.path.abspath(os.curdir)
        os.chdir(self.pre_model_dir)

        args = shlex.split(vad_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        os.chdir(current_dir)

    # def extract_ivector(self, n_jobs=10, n_threads=1, audio_dir=None, ivector_dir=None, pre_model_dir=None, debug=False):
    def extract_ivector(self, n_jobs=10, n_threads=1, audio_dir=None, ivector_dir=None, debug=False):

        audio_dir = os.path.abspath(audio_dir) if audio_dir else self.audio_dir
        ivector_dir = os.path.abspath(ivector_dir) if ivector_dir else self.ivector_dir
        # pre_model_dir = pre_model_dir if pre_model_dir else self.pre_model_dir
        extract_ivector_command = (self.pre_model_dir + "/sid/extract_ivectors.sh --cmd '$train_cmd' --nj " + str(n_jobs) +
                            " --num-threads " + str(n_threads) + " " + self.pre_model_dir + " " + 
                            audio_dir + " " + ivector_dir)

        current_dir = os.path.abspath(os.curdir)
        os.chdir(self.pre_model_dir)

        args = shlex.split(extract_ivector_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        os.chdir(current_dir)
    
    def write_trials(self, train_utt_id, test_utt_id, target=None, trials_file=None, flag=0):

        ''' train_utt_id: a list
            test_utt_id: a list
            target: a list
        '''

        trials_file = os.path.abspath(trials_file) if trials_file else self.ivector_dir + "/trials"

        if flag == 1:
            train_tmp = []
            test_tmp = []
            for i, id in enumerate(train_utt_id):
                test_tmp += test_utt_id
                train_tmp += [id for j in range(len(test_utt_id))]
            
            train_utt_id = train_tmp
            test_utt_id = test_tmp
            flag = 0
        
        if flag == 0:
            assert len(train_utt_id) == len(test_utt_id)
            n_trials = len(train_utt_id)

            if target:
                assert len(target) == len(train_utt_id)
            else:
                target = np.zeros(n_trials, dtype="<U9")
                target.fill("nontarget")

            trials_mat = np.concatenate((np.array(train_utt_id)[:, np.newaxis], 
                                         np.array(test_utt_id)[:, np.newaxis]), axis=1)
            trials_mat = np.concatenate((trials_mat, target[:, np.newaxis]), axis=1)

        np.savetxt(trials_file, trials_mat, fmt="%s")
    
    def plda_scoring(self, train_ivector_scp=None, test_ivector_scp=None, trials_file=None, scores_file=None, 
              plda=None, mean_vec=None, transform_mat=None, debug=False):

        train_ivector_scp = os.path.abspath(train_ivector_scp) if train_ivector_scp else self.ivector_dir + "/ivector.scp"
        test_ivector_scp = os.path.abspath(test_ivector_scp) if test_ivector_scp else self.ivector_dir + "/ivector.scp"
        trials_file = os.path.abspath(trials_file) if trials_file else self.ivector_dir + "/trials"
        scores_file = os.path.abspath(scores_file) if scores_file else self.ivector_dir + "/scores"
        plda = os.path.abspath(plda) if plda else self.pre_model_dir + "/plda"
        mean_vec = os.path.abspath(mean_vec) if mean_vec else self.pre_model_dir + "/mean.vec"
        transform_mat = os.path.abspath(transform_mat) if transform_mat else self.pre_model_dir + "/transform.mat"

        copy_plda_command = "ivector-copy-plda --smoothing=0.0 " + plda + " - |"
        train_ivector_command = "ark:ivector-subtract-global-mean " + mean_vec + " scp:" + train_ivector_scp + " ark:- | transform-vec " + transform_mat + " ark:- ark:- | ivector-normalize-length ark:- ark:- |"
        test_ivector_command = "ark:ivector-subtract-global-mean " + mean_vec + " scp:" + test_ivector_scp + " ark:- | transform-vec " + transform_mat + " ark:- ark:- | ivector-normalize-length ark:- ark:- |"
        trials_command = "cat " + trials_file + " | cut -d\  --fields=1,2 |"
        scores_command = ("ivector-plda-scoring --normalize-length=true " + 
                    shlex.quote(copy_plda_command) + " " +
                    shlex.quote(train_ivector_command) + " " + 
                    shlex.quote(test_ivector_command) + " " +
                    shlex.quote(trials_command) + " " +
                    scores_file)
        
        current_dir = os.path.abspath(os.curdir)
        os.chdir(self.pre_model_dir)

        args = shlex.split(scores_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        os.chdir(current_dir)
    
    def resolve_score(self, scores_file=None):

        scores_file = os.path.abspath(scores_file) if scores_file else self.ivector_dir + "/scores"
        scores_mat = np.loadtxt(scores_file, dtype=str)

        if len(scores_mat.shape) == 1:
            scores_mat = scores_mat[np.newaxis, :]

        train_utt_id = scores_mat[:, 0]
        test_utt_id = scores_mat[:, 1]
        score = (scores_mat[:, 2]).astype(np.float64)

        train_utt_id_unique = np.unique(train_utt_id)
        test_utt_id_unique = np.unique(test_utt_id)

        ''' If there is only one speaker or only one test utterance, directly return the score.
            Otherwise, return the resolved score with the shape (#utterances, #speakers)
        '''
        if train_utt_id_unique.size == 1 or test_utt_id_unique.size == 1:
            return score
        else:
            spk2score = []
            for i, utt_id in enumerate(train_utt_id_unique):
                index = np.argwhere(train_utt_id == utt_id).flatten()
                spk2score.append(score[index])

            return np.array(spk2score).T # Caution: we assume each spk have equla trials here, will raise bug if not
    
    '''
       When the audios to be scored are stored in disk (e.g., wav file), call this function
    '''
    def score_existing(self, audio_path_list, train_utt_id, spk_id_list=None, utt_id_list=None, test_utt_id=None, 
                  n_jobs=10, n_threads=1,
                  audio_dir=None, mfcc_dir=None, log_dir=None, vad_dir=None, ivector_dir=None,
                  vad_conf=None, mfcc_conf=None, target=None, trials_file=None, flag=0,
                  train_ivector_scp=None, test_ivector_scp=None, scores_file=None, 
                  plda=None, mean_vec=None, transform_mat=None, debug=False):
        
        n_audios = len(audio_path_list)
        if n_audios < n_jobs:
            n_jobs = n_audios

        utt_id_list = self.data_prepare(audio_path_list, spk_id_list=spk_id_list, utt_id_list=utt_id_list, audio_dir=audio_dir, debug=debug)
        self.make_mfcc(n_jobs=n_jobs, mfcc_conf=mfcc_conf, audio_dir=audio_dir, mfcc_dir=mfcc_dir, log_dir=log_dir, debug=debug)
        self.compute_vad(n_jobs=n_jobs, vad_conf=vad_conf, audio_dir=audio_dir, vad_dir=vad_dir, log_dir=log_dir, debug=debug)
        self.extract_ivector(n_jobs=n_jobs, n_threads=n_threads, audio_dir=audio_dir, ivector_dir=ivector_dir, debug=debug)
        
        test_utt_id = test_utt_id if test_utt_id else utt_id_list
        self.write_trials(train_utt_id, test_utt_id, target=target, trials_file=trials_file, flag=flag)
        self.plda_scoring(train_ivector_scp=train_ivector_scp, test_ivector_scp=test_ivector_scp, trials_file=trials_file, scores_file=scores_file, 
              plda=plda, mean_vec=mean_vec, transform_mat=transform_mat, debug=debug)
        
        score_array = self.resolve_score()
        return score_array
    
    '''
       When the audios to be scored are in the memory (e.g., in the form of np.array), call this function
    '''
    def score(self, audio_list, train_utt_id, spk_id_list=None, utt_id_list=None, test_utt_id=None, 
                  n_jobs=10, n_threads=1,
                  audio_dir=None, mfcc_dir=None, log_dir=None, vad_dir=None, ivector_dir=None,
                  vad_conf=None, mfcc_conf=None, target=None, trials_file=None, flag=0, 
                  train_ivector_scp=None, test_ivector_scp=None, scores_file=None, 
                  plda=None, mean_vec=None, transform_mat=None, debug=False):
        
        n_audios = len(audio_list)
        if n_audios < n_jobs:
            n_jobs = n_audios

        audio_path_list = self.write_audio(audio_list, audio_dir=audio_dir)
        utt_id_list = self.data_prepare(audio_path_list, spk_id_list=spk_id_list, utt_id_list=utt_id_list, audio_dir=audio_dir, debug=debug)
        self.make_mfcc(n_jobs=n_jobs, mfcc_conf=mfcc_conf, audio_dir=audio_dir, mfcc_dir=mfcc_dir, log_dir=log_dir, debug=debug)
        self.compute_vad(n_jobs=n_jobs, vad_conf=vad_conf, audio_dir=audio_dir, vad_dir=vad_dir, log_dir=log_dir, debug=debug)
        self.extract_ivector(n_jobs=n_jobs, n_threads=n_threads, audio_dir=audio_dir, ivector_dir=ivector_dir, debug=debug)
        
        test_utt_id = test_utt_id if test_utt_id else utt_id_list
        self.write_trials(train_utt_id, test_utt_id, target=target, trials_file=trials_file, flag=flag)
        self.plda_scoring(train_ivector_scp=train_ivector_scp, test_ivector_scp=test_ivector_scp, trials_file=trials_file, scores_file=scores_file, 
              plda=plda, mean_vec=mean_vec, transform_mat=transform_mat, debug=debug)
        
        score_array = self.resolve_score()
        return score_array  


