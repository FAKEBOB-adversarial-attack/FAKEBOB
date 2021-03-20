
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


from ivector_PLDA_kaldiHelper import ivector_PLDA_kaldiHelper
from gmm_ubm_kaldiHelper import gmm_ubm_kaldiHelper
import os
import numpy as np
import subprocess
import shlex
import pickle
import shutil

"""
This file generates speaker unique model for each speaker in the enrollment-set.
Each speaker model is a list dumped by pickle. The list contains the following items:
(1) spk id
(2) utt id of enrollment voice
(3) speaker identity location (absolute path):
data type: String
remarks: for GMM-UBM, speaker identity is a GMM obtained by updating diagonal UBM via MAP algorithm.
         for ivector-PLDA, speaker identity is an identity vector called ivector extracted by ivector-extractor (final.ie in kaldi).
(4) z-norm mean value
data type: float
remarks:
(5) z-norm std value
data type: float
remarks:
"""

''' adjustable setting
'''
n_jobs = 1
debug = False  # whether display log information from kaldi on terminal

enroll_dir = "./data/enrollment-set"  # voice data for enrollment
z_norm_dir = "./data/z-norm-set"  # voice data for z norm
pre_model_dir = "./pre-models"
model_dir = "./model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

audio_dir = os.path.abspath("./audio-build-model-iv")
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)
mfcc_dir = os.path.abspath("./mfcc-build-model-iv")
if not os.path.exists(mfcc_dir):
    os.makedirs(mfcc_dir)
log_dir = os.path.abspath("./log-build-model-iv")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
ivector_dir = os.path.abspath("./ivector-build-model-iv")
if not os.path.exists(ivector_dir):
    os.makedirs(ivector_dir)

audio_dir_gmm = os.path.abspath("./audio-build-model-gmm")
if not os.path.exists(audio_dir_gmm):
    os.makedirs(audio_dir_gmm)
mfcc_dir_gmm = os.path.abspath("./mfcc-build-model-gmm")
if not os.path.exists(mfcc_dir_gmm):
    os.makedirs(mfcc_dir_gmm)
log_dir_gmm = os.path.abspath("./log-build-model-gmm")
if not os.path.exists(log_dir_gmm):
    os.makedirs(log_dir_gmm)
score_dir = os.path.abspath("./score-build-model-gmm")
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

trials = ivector_dir + "/trials"
scores_file = ivector_dir + "/scores"
ivector_scp = ivector_dir + "/ivector.scp"
feats_scp = audio_dir + "/feats.scp"
vad_scp = audio_dir + "/vad.scp"

audio_iter = os.listdir(enroll_dir)
enroll_utt_id = []
enroll_spk_id = []
enroll_utt_path = []
for i, audio_name in enumerate(audio_iter):
    utt_id = audio_name.split(".")[0]
    spk_id = utt_id.split("-")[0]
    path = os.path.join(enroll_dir, audio_name)
    enroll_utt_path.append(path)
    enroll_utt_id.append(utt_id)
    enroll_spk_id.append(spk_id)

audio_iter = os.listdir(z_norm_dir)
z_norm_utt_id = []
z_norm_spk_id = []
z_norm_utt_path = []
for i, audio_name in enumerate(audio_iter):
    utt_id = audio_name.split(".")[0]
    spk_id = utt_id.split("-")[0]
    path = os.path.join(z_norm_dir, audio_name)
    z_norm_utt_path.append(path)
    z_norm_utt_id.append(utt_id)
    z_norm_spk_id.append(spk_id)

audio_path_list = (enroll_utt_path + z_norm_utt_path)
spk_id_list = (enroll_spk_id + z_norm_spk_id)
utt_id_list = (enroll_utt_id + z_norm_utt_id)

''' step 1: generate ivector identity (stored in ivector_dir) and corresponding speaker model (stored as model/XX.iv)
'''
print("----- step 1: generate ivector identity and corresponding speaker model -----")

iv_helper = ivector_PLDA_kaldiHelper(audio_dir=audio_dir, mfcc_dir=mfcc_dir, log_dir=log_dir, ivector_dir=ivector_dir)

print("--- extracting and scoring ---")
iv_helper.score_existing(audio_path_list, enroll_utt_id, spk_id_list=spk_id_list,
                         utt_id_list=utt_id_list, test_utt_id=z_norm_utt_id,
                         n_jobs=n_jobs, flag=1, debug=debug)

print("--- extracting and scoring done---")

print("--- resolve score and obtain z norm mean and std value ---")

scores_mat = np.loadtxt(scores_file, dtype=str)
train_utt_id = scores_mat[:, 0]
test_utt_id_scoring = scores_mat[:, 1]
score = scores_mat[:, 2].astype(np.float64)
train_spk_id = np.array([utt_id.split("-")[0] for utt_id in train_utt_id])
test_spk_id_scoring = np.array([utt_id.split("-")[0] for utt_id in test_utt_id_scoring])

z_norm_means = np.zeros(len(enroll_utt_id), dtype=np.float64)
z_norm_stds = np.zeros(len(enroll_utt_id), dtype=np.float64)

for i, id in enumerate(enroll_spk_id):
    index = np.argwhere(train_spk_id == id).flatten()
    mean = np.mean(score[index])
    std = np.std(score[index])
    z_norm_means[i] = mean
    z_norm_stds[i] = std

print("--- resolve score, and obtain z norm mean and std value done ---")

print("--- dump speaker unique model ---")

for i, utt_id in enumerate(enroll_utt_id):
    spk_id = enroll_spk_id[i]
    z_norm_mean = z_norm_means[i]
    z_norm_std = z_norm_stds[i]

    ivectors_utt_location = np.loadtxt(ivector_scp, dtype=str)
    ivectors_utt = ivectors_utt_location[:, 0]
    ivectors_location = ivectors_utt_location[:, 1]
    identity_location = os.path.abspath(
        ivectors_location[np.argwhere(ivectors_utt == utt_id).flatten()[0]])  # use absolute path

    spk_unique_model = [spk_id, utt_id, identity_location, z_norm_mean, z_norm_std]
    print(spk_unique_model),

    with open(model_dir + "/" + spk_id + ".iv", "wb") as writer:
        pickle.dump(spk_unique_model, writer, protocol=-1)

print("--- dump speaker unique model done ---")

print("----- step 1: generate ivector identity and corresponding speaker model done -----")

''' step 2: generate gmm identity (stored as model/XX-identity.gmm) and corrsponding speaker model (stored as model/XX.gmm)
'''
print("----- step 2: generate gmm identity and corresponding speaker model -----")

dubm = os.path.abspath(os.path.join(pre_model_dir, "final.dubm"))
delta_opts_file = os.path.join(pre_model_dir, "delta_opts")
with open(delta_opts_file, "r") as reader:
    delta_opts = reader.read()[:-1]
update_flags_str = "m"  # only update the mean vectors of gmm

print("--- obtaining gmm identity by updating ubm via MAP ---")
tmp_spk_feats_scp = audio_dir + "/feats_spk.scp"
tmp_spk_vad_scp = audio_dir + "/vad_spk.scp"
tmp_spk_acc_file = audio_dir + "/gmm_map_acc.acc"

feats_utt_location = np.loadtxt(feats_scp, dtype=str)
feats_utt = feats_utt_location[:, 0]
feats_location = feats_utt_location[:, 1]
vad_utt_location = np.loadtxt(vad_scp, dtype=str)
vad_utt = vad_utt_location[:, 0]
vad_location = vad_utt_location[:, 1]

for spk_id, utt_id in zip(enroll_spk_id, enroll_utt_id):
    index = np.argwhere(feats_utt == utt_id).flatten()[0]
    location = feats_location[index]
    spk_feats_scp_content = utt_id + " " + location + "\n"
    with open(tmp_spk_feats_scp, "w") as writer:
        writer.write(spk_feats_scp_content)

    index = np.argwhere(vad_utt == utt_id).flatten()[0]
    location = vad_location[index]
    spk_vad_scp_content = utt_id + " " + location + "\n"
    with open(tmp_spk_vad_scp, "w") as writer:
        writer.write(spk_vad_scp_content)

    add_deltas = ("add-deltas " + delta_opts + " scp:" + tmp_spk_feats_scp + " ark:- |")
    apply_cmvn = "apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
    select_voiced_frame = ("select-voiced-frames ark:- scp,s,cs:" + tmp_spk_vad_scp + " ark:- |")
    feats = ("ark,s,cs:" + add_deltas + " " + apply_cmvn + " " + select_voiced_frame)

    acc_stats_command = ("gmm-global-acc-stats --binary=false --update-flags=" +
                         update_flags_str + " " +
                         dubm + " " +
                         shlex.quote(feats) + " " +
                         tmp_spk_acc_file)
    args = shlex.split(acc_stats_command)
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()

    output_model = model_dir + "/" + spk_id + "-identity.gmm"
    map_command = ("gmm-global-est-map --update-flags=" +
                   update_flags_str + " " +
                   dubm + " " +
                   tmp_spk_acc_file + " " +
                   output_model)
    args = shlex.split(map_command)
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()

    # delete all the tmp file
    os.remove(tmp_spk_feats_scp)
    os.remove(tmp_spk_vad_scp)
    os.remove(tmp_spk_acc_file)

print("--- obtaining gmm identity by updating ubm via MAP done ---")

gmm_helper = gmm_ubm_kaldiHelper(pre_model_dir=pre_model_dir, audio_dir=audio_dir_gmm,
                                 mfcc_dir=mfcc_dir_gmm, log_dir=log_dir_gmm, score_dir=score_dir)

model_path_list = []
for spk_id in enroll_spk_id:
    model_path = model_dir + "/" + spk_id + "-identity.gmm"
    model_path_list.append(model_path)

print("--- calculate z-norm mean, z-norm std ---")

# clear directory, otherwise kaldi may not keep all the audios to be scored.
if os.path.exists(audio_dir_gmm):
    shutil.rmtree(audio_dir_gmm)
if os.path.exists(mfcc_dir_gmm):
    shutil.rmtree(mfcc_dir_gmm)
if os.path.exists(log_dir_gmm):
    shutil.rmtree(log_dir_gmm)
if os.path.exists(score_dir):
    shutil.rmtree(score_dir)

if not os.path.exists(audio_dir_gmm):
    os.makedirs(audio_dir_gmm)
if not os.path.exists(mfcc_dir_gmm):
    os.makedirs(mfcc_dir_gmm)
if not os.path.exists(log_dir_gmm):
    os.makedirs(log_dir_gmm)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

''' calculate z-norm and z-std. Note that z-norm is only used in CSI. IN SV and OSI, we use UBM norm.
'''
score_array = gmm_helper.score_existing(model_path_list, z_norm_utt_path, n_jobs=n_jobs, debug=debug)
z_norm_means = np.mean(score_array, axis=0).flatten()
z_norm_stds = np.std(score_array, axis=0).flatten()

print("--- calculate z-norm mean, z-norm std done ---")

print(" --- dump speaker unique model --- ")
for i, spk_id in enumerate(enroll_spk_id):
    utt_id = enroll_utt_id[i]
    identity_location = os.path.abspath(model_dir + "/" + spk_id + "-identity.gmm")
    z_norm_mean = z_norm_means[i]
    z_norm_std = z_norm_stds[i]

    spk_unique_model = [spk_id, utt_id, identity_location, z_norm_mean, z_norm_std]

    print(spk_unique_model),

    with open(model_dir + "/" + spk_id + ".gmm", "wb") as writer:
        pickle.dump(spk_unique_model, writer, protocol=-1)

print(" --- dump speaker unique model done --- ")

print("----- step 2: generate gmm identity and corresponding speaker model done -----")
