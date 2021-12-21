
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import argparse
import os
import pickle
import random

import numpy as np
from scipy.io.wavfile import read, write

from FAKEBOB import FakeBob
from gmm_ubm_CSI import gmm_CSI
from gmm_ubm_OSI import gmm_OSI
from gmm_ubm_SV import gmm_SV
from ivector_PLDA_CSI import iv_CSI
from ivector_PLDA_OSI import iv_OSI
from ivector_PLDA_SV import iv_SV

IV = "iv"
GMM = "gmm"
OSI = "OSI"
CSI = "CSI"
SV = "SV"
UNTARGETED = "untargeted"
TARGETED = "targeted"

bits_per_sample = 16
fs = 16000
model_dir = "./model"
pre_model_dir = "pre-models"
test_dir = "./data/test-set"
illegal_dir = "./data/illegal-set"

def load_model(spk_id_list, architecture, task, threshold, id):

    iv_model_paths = [os.path.join(model_dir, spk_id + ".iv") for spk_id in spk_id_list]
    gmm_model_paths = [os.path.join(model_dir, spk_id + ".gmm") for spk_id in spk_id_list]

    iv_model_list = []
    gmm_model_list = []

    for path in iv_model_paths:
        with open(path, "rb") as reader:
            model = pickle.load(reader)
            iv_model_list.append(model)
    for path in gmm_model_paths:
        with open(path, "rb") as reader:
            model = pickle.load(reader)
            gmm_model_list.append(model)

    ubm = os.path.join(pre_model_dir, "final.dubm")

    if architecture == IV:

        if task == OSI:

            model = iv_OSI(id, iv_model_list, threshold=threshold)

        elif task == CSI:

            model = iv_CSI(id, iv_model_list)

        else:

            model = iv_SV(id, iv_model_list[0], threshold=threshold)

    else:

        if task == OSI:

            model = gmm_OSI(id, gmm_model_list, ubm, threshold=threshold)

        elif task == CSI:

            model = gmm_CSI(id, gmm_model_list)

        else:

            model = gmm_SV(id, gmm_model_list[0], ubm, threshold=threshold)

    return model

def loadData(task, attack_type, model, spk_id_list, n_jobs=10, debug=False):

    audio_names = []
    adver_audio_paths = []
    checkpoint_paths = []

    if task == CSI:

        spk_ids = np.array(model.spk_ids)
        data_path = test_dir
        audio_list = []
        true_label_list = []
        spk_iter = os.listdir(data_path)
        for spk_id in spk_iter:

            true_label = np.argwhere(spk_ids == spk_id).flatten()[0]

            spk_dir = os.path.join(data_path, spk_id)
            audio_iter = os.listdir(spk_dir)

            adver_audio_spk_dir = os.path.join(adver_audio_dir, spk_id)
            if not os.path.exists(adver_audio_spk_dir):
                os.makedirs(adver_audio_spk_dir)
            
            checkpoint_spk_dir = os.path.join(checkpoint_dir, spk_id)
            if not os.path.exists(checkpoint_spk_dir):
                os.makedirs(checkpoint_spk_dir)

            for audio_name in audio_iter:

                audio_names.append(audio_name)
                adver_audio_paths.append(os.path.join(os.path.join(adver_audio_dir, spk_id), audio_name))
                checkpoint_paths.append(os.path.join(os.path.join(checkpoint_dir, spk_id), audio_name.split(".")[0] + ".cp"))

                audio_path = os.path.join(spk_dir, audio_name)
                _, audio = read(audio_path)
                audio = audio / (2 ** (bits_per_sample - 1))
                audio_list.append(audio)
                true_label_list.append(true_label)
        
        # skip those wrongly classified
        decisions, _ = model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        preserve_index = np.argwhere(np.array(decisions) == true_label_list).flatten()
        audio_list = [audio_list[index] for index in preserve_index]
        true_label_list = [true_label_list[index] for index in preserve_index]
        audio_names = [audio_names[index] for index in preserve_index]
        adver_audio_paths = [adver_audio_paths[index] for index in preserve_index]
        checkpoint_paths = [checkpoint_paths[index] for index in preserve_index]

        if attack_type == UNTARGETED:

            return audio_list, true_label_list, None, audio_names, adver_audio_paths, checkpoint_paths
        
        audio_list_targeted = []
        audio_names_targeted = []
        adver_audio_paths_targeted = []
        checkpoint_paths_targeted = []
        true_label_list_targeted = []
        target_label_list = []
        for audio, true_label, audio_name, adver_audio_path, checkpoint_path in zip(audio_list, 
                                                                                    true_label_list, 
                                                                                    audio_names, 
                                                                                    adver_audio_paths, 
                                                                                    checkpoint_paths):

            for target_label in range(len(model.spk_ids)):

                if target_label == true_label:
                    continue

                audio_list_targeted.append(audio)
                audio_names_targeted.append(audio_name)
                adver_audio_paths_targeted.append(adver_audio_path.split(".")[0] + "_" + str(target_label) + ".wav")
                checkpoint_paths_targeted.append(checkpoint_path.split(".")[0] + "_" + str(target_label) + ".cp")
                true_label_list_targeted.append(true_label)
                target_label_list.append(target_label)
        
        audio_names = audio_names_targeted
        adver_audio_paths = adver_audio_paths_targeted
        checkpoint_paths = checkpoint_paths_targeted
        
        return audio_list_targeted, true_label_list_targeted, target_label_list, audio_names, adver_audio_paths, checkpoint_paths

    elif task == OSI:

        data_path = illegal_dir
        audio_list = []
        spk_iter = os.listdir(data_path)
        for spk_id in spk_iter:

            spk_dir = os.path.join(data_path, spk_id)
            audio_iter = os.listdir(spk_dir)

            adver_audio_spk_dir = os.path.join(adver_audio_dir, spk_id)
            if not os.path.exists(adver_audio_spk_dir):
                os.makedirs(adver_audio_spk_dir)
            
            checkpoint_spk_dir = os.path.join(checkpoint_dir, spk_id)
            if not os.path.exists(checkpoint_spk_dir):
                os.makedirs(checkpoint_spk_dir)
            
            for audio_name in audio_iter:

                audio_names.append(audio_name)
                adver_audio_paths.append(os.path.join(os.path.join(adver_audio_dir, spk_id), audio_name))
                checkpoint_paths.append(os.path.join(os.path.join(checkpoint_dir, spk_id), audio_name.split(".")[0] + ".cp"))

                audio_path = os.path.join(spk_dir, audio_name)
                _, audio = read(audio_path)
                audio = audio / (2 ** (bits_per_sample - 1))
                audio_list.append(audio)
        
        # skip those far audios
        decisions, _ = model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        preserve_index = np.argwhere(np.array(decisions) == -1).flatten()
        audio_list = [audio_list[index] for index in preserve_index]
        audio_names = [audio_names[index] for index in preserve_index]
        adver_audio_paths = [adver_audio_paths[index] for index in preserve_index]
        checkpoint_paths = [checkpoint_paths[index] for index in preserve_index]

        if attack_type == UNTARGETED:

            return audio_list, None, None, audio_names, adver_audio_paths, checkpoint_paths
        
        audio_list_targeted = []
        audio_names_targeted = []
        adver_audio_paths_targeted = []
        checkpoint_paths_targeted = []
        target_label_list = []
        for audio, audio_name, adver_audio_path, checkpoint_path in zip(audio_list,
                                                                        audio_names,
                                                                        adver_audio_paths,
                                                                        checkpoint_paths):

            for target_label in range(len(model.spk_ids)):

                audio_list_targeted.append(audio)
                audio_names_targeted.append(audio_name)
                adver_audio_paths_targeted.append(adver_audio_path.split(".")[0] + "_" + str(target_label) + ".wav")
                checkpoint_paths_targeted.append(checkpoint_path.split(".")[0] + "_" + str(target_label) + ".cp")
                target_label_list.append(target_label)
        
        audio_names = audio_names_targeted
        adver_audio_paths = adver_audio_paths_targeted
        checkpoint_paths = checkpoint_paths_targeted

        return audio_list_targeted, None, target_label_list, audio_names, adver_audio_paths, checkpoint_paths
    
    else: # SV

        audio_list = []
        data_path = illegal_dir
        spk_iter = os.listdir(data_path)
        for spk_id in spk_iter:

            spk_dir = os.path.join(data_path, spk_id)
            audio_iter = os.listdir(spk_dir)

            adver_audio_spk_dir = os.path.join(adver_audio_dir, spk_id)
            if not os.path.exists(adver_audio_spk_dir):
                os.makedirs(adver_audio_spk_dir)
            checkpoint_spk_dir = os.path.join(checkpoint_dir, spk_id)
            if not os.path.exists(checkpoint_spk_dir):
                os.makedirs(checkpoint_spk_dir)
            
            for audio_name in audio_iter:

                audio_names.append(audio_name)
                adver_audio_paths.append(os.path.join(os.path.join(adver_audio_dir, spk_id), audio_name))
                checkpoint_paths.append(os.path.join(os.path.join(checkpoint_dir, spk_id), audio_name.split(".")[0] + ".cp"))

                audio_path = os.path.join(spk_dir, audio_name)
                _, audio = read(audio_path)
                audio = audio / (2 ** (bits_per_sample - 1))
                audio_list.append(audio)
        
        # skip those far audios

        decisions, _ = model.make_decisions(audio_list, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        preserve_index = np.argwhere(np.array(decisions) == -1).flatten()
        audio_list = [audio_list[index] for index in preserve_index]
        audio_names = [audio_names[index] for index in preserve_index]
        adver_audio_paths = [adver_audio_paths[index] for index in preserve_index]
        checkpoint_paths = [checkpoint_paths[index] for index in preserve_index]

        return audio_list, None, None, audio_names, adver_audio_paths, checkpoint_paths

def main(spk_id_list, architecture, task, threshold, attack_type, adver_thresh,
         epsilon, max_iter, max_lr, min_lr, samples_per_draw, sigma,
         momentum, plateau_length, plateau_drop, n_jobs, debug):
    
    id = architecture + "-" + task + "-" + attack_type
    global adver_audio_dir
    adver_audio_dir = os.path.join("adversarial-audio", id)
    global checkpoint_dir
    checkpoint_dir = os.path.join("checkpoint", id)

    if task == SV:

        adver_audio_dir = os.path.join(adver_audio_dir, spk_id_list[0])
        checkpoint_dir = os.path.join(checkpoint_dir, spk_id_list[0])

    if not os.path.exists(adver_audio_dir):
        os.makedirs(adver_audio_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    '''
    load model
    '''
    print('------ load model ------')
    model = load_model(spk_id_list, architecture, task, threshold, id)
    print('------ load model done ------')
    
    '''
    load data
    '''
    print('------ load data ------')

    audio_list, true_label_list, target_label_list, \
    audio_names, adver_audio_paths, checkpoint_paths = loadData(task, attack_type, model, spk_id_list, n_jobs=n_jobs, debug=debug)
    total_cnt = len(audio_list)

    print('------ load data done, total num: %d ------' %total_cnt)
    
    success_cnt = 0

    print("----- generate adversarial voices -----")

    fake_bob = FakeBob(task, attack_type, model, adver_thresh=adver_thresh, epsilon=epsilon, max_iter=max_iter,
                         max_lr=max_lr, min_lr=min_lr, samples_per_draw=samples_per_draw, sigma=sigma, momentum=momentum, 
                         plateau_length=plateau_length, plateau_drop=plateau_drop)

    if task == CSI:

        if attack_type == TARGETED:

            for audio, true_label, \
                target_label, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, true_label_list, 
                                                        target_label_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                
                print("--- %s, %s, %s, audio name:%s, true spk:%s, target spk:%s ---" %(architecture, task, attack_type, audio_name, model.spk_ids[true_label], model.spk_ids[target_label]))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, target=target_label, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1
            
        else:

            for audio, true_label, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, true_label_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                
                print("--- %s, %s, %s, audio name:%s, true spk:%s ---" %(architecture, task, attack_type, audio_name, model.spk_ids[true_label]))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, true=true_label, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1


    elif task == OSI:
        
        # first estimates the threshold
        audio = audio_list[np.random.choice(total_cnt, 1)[0]] # randomly choose an audio to estimate the threshold
        threshold_estimated, _, _ = fake_bob.estimate_threshold(audio, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)

        # threshold_estimated = 2.10 # iv-OSI
        # threshold_estimated = 0.23 # gmm-OSI

        if attack_type == TARGETED:

            for audio, target_label, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, 
                                                        target_label_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                
                print("--- %s, %s, %s, audio name:%s, target spk:%s ---" %(architecture, task, attack_type, audio_name, model.spk_ids[target_label]))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, threshold=threshold_estimated, target=target_label, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1
        
        else:

            for audio, audio_name, \
                adver_audio_path, checkpoint_path in zip(audio_list, audio_names, 
                                                        adver_audio_paths, checkpoint_paths):
                
                print("--- %s, %s, %s, audio name:%s ---" %(architecture, task, attack_type, audio_name))
                adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, threshold=threshold_estimated, fs=fs, 
                                                            bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                
                write(adver_audio_path, fs, adver_audio)
                if success_flag == 1:
                    success_cnt += 1

    else:
        
        audio = audio_list[np.random.choice(total_cnt, 1)[0]] # randomly choose an audio to estimate the threshold
        threshold_estimated, _, _ = fake_bob.estimate_threshold(audio, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        
        # threshold_estimated = 1.83 # iv-SV
        # threshold_estimated = 0.15 # gmm-SV

        for audio, audio_name, \
            adver_audio_path, checkpoint_path in zip(audio_list, audio_names, 
                                                    adver_audio_paths, checkpoint_paths):
                
            print("--- %s, %s, %s, audio name:%s ---" %(architecture, task, attack_type, audio_name))
            adver_audio, success_flag = fake_bob.attack(audio, checkpoint_path, threshold=threshold_estimated, fs=fs, 
                                                        bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                
            write(adver_audio_path, fs, adver_audio)
            if success_flag == 1:
                success_cnt += 1

    print('------ attack successful rate %d ------' %(success_cnt * 100 / total_cnt))
    print("----- generate adversarial voices done -----")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--speaker_id", "-spk_id", nargs="+", type=str)

    parser.add_argument("--architecture", "-archi", default=GMM, choices=[GMM, IV], type=str)
    parser.add_argument("--task", "-task", default=OSI, choices=[OSI, CSI, SV], type=str)
    parser.add_argument("--attack_type", "-type", default=TARGETED, choices=[UNTARGETED, TARGETED], type=str) # obmit when task is SV

    parser.add_argument("--adver_thresh", "-adver", default=0., type=float)
    parser.add_argument("--epsilon", "-epsilon", default=0.002, type=float)
    parser.add_argument("--max_iter", "-max_iter", default=1000, type=int)
    parser.add_argument("--max_lr", "-max_lr", default=0.001, type=float)
    parser.add_argument("--min_lr", "-min_lr", default=1e-6, type=float)
    parser.add_argument("--samples_per_draw", "-samples", default=50, type=int)
    parser.add_argument("--sigma", "-sigma", default=0.001, type=float)
    parser.add_argument("--momentum", "-momentum", default=0.9, type=float)
    parser.add_argument("--plateau_length", "-plateau_length", default=5, type=int)
    parser.add_argument("--plateau_drop", "-plateau_drop", default=2.0, type=float)

    # parser.add_argument("--n_jobs", "-nj", default=10, type=int)
    parser.add_argument("--n_jobs", "-nj", default=1, type=int)
    # parser.add_argument("--debug", "-debug", default=False, type=bool)
    parser.add_argument("--debug", "-debug", default="f", type=str, choices=["t", "f"]) # "f" for False, "t" for True

    parser.add_argument("--threshold", "-thresh", default=0., type=float) # only meaningful for OSI and SV task

    args = parser.parse_args()

    spk_id_list = args.speaker_id

    architecture = args.architecture
    task = args.task
    attack_type = args.attack_type
    if task == SV:
        attack_type = TARGETED
        spk_id_list = spk_id_list[0:1] # SV only support one enrolled speakers
    
    adver_thresh = args.adver_thresh
    epsilon = args.epsilon
    max_iter = args.max_iter
    max_lr = args.max_lr
    min_lr = args.min_lr
    samples_per_draw = args.samples_per_draw
    sigma = args.sigma
    momentum = args.momentum
    plateau_length = args.plateau_length
    plateau_drop = args.plateau_drop

    n_jobs = args.n_jobs
    debug = args.debug
    if debug == "f":
        debug = False
    else:
        debug = True

    threshold = args.threshold

    main(spk_id_list, architecture, task, threshold, attack_type, adver_thresh,
         epsilon, max_iter, max_lr, min_lr, samples_per_draw, sigma,
         momentum, plateau_length, plateau_drop, n_jobs, debug)
