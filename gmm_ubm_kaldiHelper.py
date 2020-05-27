
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import numpy as np
import os
from scipy.io.wavfile import write
import subprocess
import shlex

class gmm_ubm_kaldiHelper(object):

    def __init__(self, pre_model_dir="pre-models", audio_dir=None, mfcc_dir=None, log_dir=None, score_dir=None):

        self.pre_model_dir = os.path.abspath(pre_model_dir)
        self.conf_dir = os.path.join(self.pre_model_dir, "conf")
        audio_dir = audio_dir if audio_dir else "audio"
        self.audio_dir = os.path.abspath(audio_dir)

        mfcc_dir = mfcc_dir if mfcc_dir else "mfcc"
        self.mfcc_dir = os.path.abspath(mfcc_dir)

        log_dir = log_dir if log_dir else "log"
        self.log_dir = os.path.abspath(log_dir)

        score_dir = score_dir if score_dir else "score"
        self.score_dir = os.path.abspath(score_dir)

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

        n_audios = len(audio_path_list)
        spk_id_list = spk_id_list if spk_id_list else [("0000" + str(i+1))[-5:] for i in range(n_audios)] # this may cause a bug if there are more than 100000 audios in audio_list
        utt_id_list = utt_id_list if utt_id_list else [spk_id + "-{}".format(1) for spk_id in spk_id_list]

        # wav.scp, utt2spk
        wav_file = self.audio_dir + "/wav.scp"
        utt2spk_file = self.audio_dir + "/utt2spk"
        wav_mat = np.concatenate((np.array(utt_id_list)[:, np.newaxis], np.array(audio_path_list)[:, np.newaxis]), axis=1)
        np.savetxt(wav_file, wav_mat, fmt="%s")
        utt2spk_mat = np.concatenate((np.array(utt_id_list)[:, np.newaxis], np.array(spk_id_list)[:, np.newaxis]), axis=1)
        np.savetxt(utt2spk_file, utt2spk_mat, fmt="%s")

        # spk2utt
        spk_id_vector = utt2spk_mat[:, 1]
        spk_id_unique_vector = np.unique(spk_id_vector)
        spk2utt_list = []
        for spk_id in spk_id_unique_vector:
            utt_index = np.argwhere(spk_id_vector == spk_id).flatten()
            utts = ""
            for index in utt_index:
                utts += (utt2spk_mat[index, 0] + " ")
            utts.rstrip()
            spk2utt_list.append(utts)

        spk2utt_mat = np.concatenate((spk_id_unique_vector[:, np.newaxis], np.array(spk2utt_list)[:, np.newaxis]), axis=1)
        spk2utt_file = self.audio_dir + "/spk2utt"
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

        vad_command = (self.pre_model_dir + "/sid/compute_vad_decision.sh --nj " + str(n_jobs) +
                        " --cmd '$train_cmd' --vad-config " + vad_conf + " " +
                        audio_dir + " " + log_dir + " " + vad_dir)
        
        current_dir = os.path.abspath(os.curdir)
        os.chdir(self.pre_model_dir)

        args = shlex.split(vad_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        os.chdir(current_dir)
    
    def get_frames_likes(self, model_path_list, n_jobs=10, debug=False):

        cmd = self.pre_model_dir + "/utils/run.pl"
        n_threads = 1
        get_frame_likes_log = self.log_dir + "/get_frame_likes.JOB.log"
        adverage="--average=true"

        # split data for parallel running
        current_dir = os.path.abspath(os.curdir)
        os.chdir(self.pre_model_dir)

        split_data_command = self.pre_model_dir + "/utils/split_data.sh" + " " + self.audio_dir + " " + str(n_jobs)
        args = shlex.split(split_data_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        os.chdir(current_dir)

        sdata = self.audio_dir + "/split" + str(n_jobs)

        delta_opts_file = os.path.join(self.pre_model_dir, "delta_opts")
        with open(delta_opts_file, "r") as reader:
            delta_opts = reader.read()[:-1]

        add_deltas = ("add-deltas " + delta_opts + " scp:" + sdata + "/JOB/feats.scp" + " ark:- |")
        apply_cmvn = "apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
        select_voiced_frame = ("select-voiced-frames ark:- scp,s,cs:"  + sdata + "/JOB/vad.scp" + " ark:- |")
        feats = ("ark,s,cs:" + add_deltas + " " + apply_cmvn + " " + select_voiced_frame)

        job_scores_file = self.score_dir + "/score.JOB"

        for i, model_path in enumerate(model_path_list):

            current_dir = os.path.abspath(os.curdir)
            os.chdir(self.pre_model_dir)
            get_frames_likes_command = ("gmm-global-get-frame-likes " + 
                                        adverage + " " + model_path + " " + 
                                        shlex.quote(feats) + " ark,t:" + job_scores_file)
            # cmd_command = (cmd + " --num-threads " + str(n_threads) + 
            #                " JOB=1:" + str(n_jobs) + " " + 
            #                get_frame_likes_log + " " + 
            #                shlex.quote(get_frames_likes_command) + " || exit 1;")
            cmd_command = (cmd + " --num-threads " + str(n_threads) + 
                           " JOB=1:" + str(n_jobs) + " " + 
                           get_frame_likes_log + " " + 
                           get_frames_likes_command + " || exit 1;")
            
            args = shlex.split(cmd_command)
            p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            p.wait()
            os.chdir(current_dir)

            scores_file = self.score_dir + "/" + str(i+1) + ".score"
            content = []
            for job_id in range(1, n_jobs+1):

                job_score_file = self.score_dir + "/score." + str(job_id)
                with open(job_score_file, "r") as reader:
                    content += reader.readlines()
                
                os.remove(job_score_file)
            
            with open(scores_file, "w") as writer:
                writer.writelines(content)

    def resolce_scores(self, model_path_list):

        score_array = []
        for i in range(1, len(model_path_list) + 1):
            scores_file = self.score_dir + "/" + str(i) + ".score"
            score = np.loadtxt(scores_file, usecols=1) # (n_audios, ) or a scalar(if only one audio)
            score_array.append(score)
        score_array = np.array(score_array) # (n_models, n_audios) or (n_models, ) (if only one audio)
        if len(score_array.shape) == 1:
            score_array = score_array[:, np.newaxis]
        score_array = score_array.T # (n_audios, n_models)

        return score_array

    def score_existing(self, model_path_list, audio_path_list, spk_id_list=None, utt_id_list=None, 
                       fs=16000, n_jobs=10, debug=False, bits_per_sample=16, mfcc_conf=None):
        

        ''' make model path being absolute path
        '''
        model_path_list = [os.path.abspath(path) for path in model_path_list]

        n_audios = len(audio_path_list)
        if n_jobs > n_audios:
            n_jobs = n_audios
        
        self.data_prepare(audio_path_list, utt_id_list=utt_id_list, spk_id_list=spk_id_list, debug=debug)
        self.make_mfcc(n_jobs=n_jobs, mfcc_conf=mfcc_conf, debug=debug)
        self.compute_vad(n_jobs=n_jobs, debug=debug)
        self.get_frames_likes(model_path_list, n_jobs=n_jobs, debug=debug)
        score_array = self.resolce_scores(model_path_list)

        return score_array

    def score(self, model_path_list, audio_list, fs=16000, n_jobs=10, debug=False, bits_per_sample=16, mfcc_conf=None):

        ''' model_path_list is a list object and each item is the location of a gmm identity (XXX-identity.gmm)
        '''

        ''' make model path being absolute path
        '''
        model_path_list = [os.path.abspath(path) for path in model_path_list]

        n_audios = len(audio_list)
        if n_jobs > n_audios:
            n_jobs = n_audios
         
        audio_path_list = self.write_audio(audio_list, fs=fs)

        self.data_prepare(audio_path_list, debug=debug)
        self.make_mfcc(n_jobs=n_jobs, mfcc_conf=mfcc_conf, debug=debug)
        self.compute_vad(n_jobs=n_jobs, debug=debug)
        self.get_frames_likes(model_path_list, n_jobs=n_jobs, debug=debug)
        score_array = self.resolce_scores(model_path_list)

        return score_array