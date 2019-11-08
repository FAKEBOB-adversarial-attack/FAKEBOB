
import numpy as np
import os
from scipy.io.wavfile import write
import subprocess
import shlex

class kaldiHelper(object):

    def __init__(self, root_dir, audio_dir, mfcc_dir, log_dir, score_dir, system_dir):

        self.root_dir = root_dir
        self.audio_dir = audio_dir
        self.mfcc_dir = mfcc_dir
        self.log_dir = log_dir
        self.score_dir = score_dir
        self.system_dir = system_dir

    def score(self, model_path_list, audio_list, fs=16000, n_jobs=28, debug=False, bits_per_sample=16, mfcc_conf=None):

        """ write audios in audio_list to self.audio.dir, generate wav.scp, spk2utt, utt2spk
        """
        wav_file = self.audio_dir + "/wav.scp"
        utt2spk_file = self.audio_dir + "/utt2spk"
        spk2utt_file = self.audio_dir + "/spk2utt"
        wav_writer = open(wav_file, "w")
        utt2spk_writer = open(utt2spk_file, "w")
        spk2utt_writer = open(spk2utt_file, "w")
        for i, audio in enumerate(audio_list):
            spk_id = ("0000" + str(i))[-5:] # this may cause a bug if there are more than 100000 audios in audio_list
            utt_id = spk_id + "-0"
            audio_path = self.audio_dir + "/" + utt_id + ".wav"
            write(audio_path, fs, audio)
            wav_line = utt_id + " " + audio_path + "\n"
            wav_writer.write(wav_line)
            utt2spk_line = utt_id + " " + spk_id + "\n"
            utt2spk_writer.write(utt2spk_line)
            spk2utt_line = spk_id + " " + utt_id + "\n"
            spk2utt_writer.write(spk2utt_line)
        
        spk2utt_writer.close()
        utt2spk_writer.close()
        wav_writer.close()
        
        if len(audio_list) < n_jobs:
            n_jobs = len(audio_list)

        #n_jobs = len(audio_list)

        """ make mfcc
        """
        mfcc_conf = mfcc_conf if mfcc_conf else self.system_dir + "/conf/mfcc.conf"
        extract_mfcc_command = (self.root_dir + "/steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config " + 
                                mfcc_conf + " --nj " + str(n_jobs) + " --cmd '$train_cmd' " + self.audio_dir + " " + 
                                self.log_dir + " " + self.mfcc_dir)
        args = shlex.split(extract_mfcc_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        """ compute_vad
        """
        vad_conf = self.system_dir + "/conf/vad.conf"
        vad_command = (self.root_dir + "/sid/compute_vad_decision.sh --nj " + str(n_jobs) + 
                        " --cmd '$train_cmd' --vad-config " + vad_conf + " " + 
                        self.audio_dir + " " + self.log_dir + " " + self.mfcc_dir)
        args = shlex.split(vad_command)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        """ score
        """
        n_models = len(model_path_list)
        model_path_str=""
        for model_path in model_path_list:
            model_path_str += (model_path + ";")
        model_path_str = model_path_str[:-1]

        delta_opts = self.system_dir + "/delta_opts"
        score_command = (self.root_dir + "/kaldiHelper.sh " + str(n_jobs) + " " + str(n_models) + " " 
                        + model_path_str + " " + self.audio_dir + " " + self.log_dir + " " + self.score_dir 
                        + " " + delta_opts)
        args = shlex.split(score_command)
        #print(args)
        p = subprocess.Popen(args) if debug else subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait()

        """ resolve score
        """
        score_array = []
        for i in range(1, len(model_path_list) + 1):
            scores_file = self.score_dir + "/" + "scores." + str(i)
            score = np.loadtxt(scores_file, usecols=1) # (n_audios, ) or a scalar(if only one audio)
            #print(score)
            score_array.append(score)
        score_array = np.array(score_array) # (n_models, n_audios) or (n_models, ) (if only one audio)
        if len(score_array.shape) == 1:
            score_array = score_array[:, np.newaxis]
        score_array = score_array.T # (n_audios, n_models)

        #print(score_array)

        return score_array



        
        
        















