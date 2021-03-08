
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


import numpy as np
import os
from ivector_PLDA_OSI import iv_OSI
from ivector_PLDA_CSI import iv_CSI
from ivector_PLDA_SV import iv_SV
from gmm_ubm_OSI import gmm_OSI
from gmm_ubm_CSI import gmm_CSI
from gmm_ubm_SV import gmm_SV
from scipy.io.wavfile import read
import pickle

debug = False
n_jobs = 4

test_dir = "./data/test-set"
illegal_dir = "./data/illegal-set"

model_dir = "model"
spk_id_list = ["1580", "2830", "4446", "5142", "61"]  # Change to your own spk ids !!!!
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

pre_model_dir = "pre-models"
ubm = os.path.join(pre_model_dir, "final.dubm")


def set_threshold(score_target, score_untarget):

    if not isinstance(score_target, np.ndarray):
        score_target = np.array(score_target)
    if not isinstance(score_untarget, np.ndarray):
        score_untarget = np.array(score_untarget)

    n_target = score_target.size
    n_untarget = score_untarget.size

    final_threshold = 0.
    min_difference = np.infty
    final_far = 0.
    final_frr = 0.
    for candidate_threshold in score_target:

        frr = np.argwhere(score_target < candidate_threshold).flatten().size * 100 / n_target
        far = np.argwhere(score_untarget >= candidate_threshold).flatten().size * 100 / n_untarget
        difference = np.abs(frr - far)
        if difference < min_difference:
            final_threshold = candidate_threshold
            final_far = far
            final_frr = frr
            min_difference = difference

    return final_threshold, final_frr, final_far


''' Test for ivector-PLDA-based CSI
'''
group_id = "test-iv-CSI"
iv_csi_model = iv_CSI(group_id, iv_model_list)
spk_ids = np.array(iv_csi_model.spk_ids)

audio_list = []
target_label_list = []
spk_iter = os.listdir(test_dir)
for spk_id in spk_iter:

    target_label = np.argwhere(spk_ids == spk_id).flatten()[0]

    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        audio_list.append(audio)
        target_label_list.append(target_label)

decisions, _ = iv_csi_model.make_decisions(audio_list, debug=debug, n_jobs=n_jobs)
decisions = np.array(decisions)
target_label_list = np.array(target_label_list)
correct_cnt = np.argwhere(decisions == target_label_list).flatten().size
acc = correct_cnt * 100 / decisions.size
print("----- Test of ivector-PLDA-based CSI, result ---> Accuracy:%f ----- " % (acc)),





''' Test for gmm-ubm-based CSI
'''
group_id = "test-gmm-CSI"
gmm_csi_model = gmm_CSI(group_id, gmm_model_list)
spk_ids = np.array(gmm_csi_model.spk_ids)

audio_list = []
target_label_list = []
spk_iter = os.listdir(test_dir)
for spk_id in spk_iter:

    target_label = np.argwhere(spk_ids == spk_id).flatten()[0]

    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        audio_list.append(audio)
        target_label_list.append(target_label)

decisions, _ = gmm_csi_model.make_decisions(audio_list, debug=debug, n_jobs=n_jobs)
decisions = np.array(decisions)
target_label_list = np.array(target_label_list)
correct_cnt = np.argwhere(decisions == target_label_list).flatten().size
acc = correct_cnt * 100 / decisions.size
print("----- Test of gmm-ubm-based CSI, result ---> Accuracy:%f ----- " % (acc)),


''' Test for ivector-PLDA-based SV
'''
spk_iter = os.listdir(illegal_dir)
illegal_audio_list = []
for spk_id in spk_iter:
    spk_dir = os.path.join(illegal_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        illegal_audio_list.append(audio)

score_target = []
score_untarget = []

for model in iv_model_list:

    spk_id = model[0]
    spk_id_extra = "test-iv-SV-" + spk_id
    iv_sv_model = iv_SV(spk_id_extra, model)

    audio_list = []
    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        audio_list.append(audio)

    _, scores = iv_sv_model.make_decisions(audio_list, n_jobs=n_jobs, debug=debug)
    score_target += [score for score in scores]

    _, scores = iv_sv_model.make_decisions(illegal_audio_list, n_jobs=n_jobs, debug=debug)
    score_untarget += [score for score in scores]

threshold, frr, far = set_threshold(score_target, score_untarget)
print("----- Test of ivector-PLDA-based SV, result ---> threshold: %f FRR: %f, FAR: %f" % (threshold, frr, far))





''' Test for gmm-ubm-based SV
'''
spk_iter = os.listdir(illegal_dir)
illegal_audio_list = []
for spk_id in spk_iter:
    spk_dir = os.path.join(illegal_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        illegal_audio_list.append(audio)

score_target = []
score_untarget = []
for model in gmm_model_list:

    spk_id = model[0]
    spk_id_extra = "test-gmm-SV-" + spk_id
    gmm_sv_model = gmm_SV(spk_id_extra, model, ubm)

    audio_list = []
    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        audio_list.append(audio)

    _, scores = gmm_sv_model.make_decisions(audio_list, n_jobs=n_jobs, debug=debug)
    score_target += [score for score in scores]

    _, scores = gmm_sv_model.make_decisions(illegal_audio_list, n_jobs=n_jobs, debug=debug)
    score_untarget += [score for score in scores]

threshold, frr, far = set_threshold(score_target, score_untarget)
print("----- Test of gmm-ubm-based SV, result ---> threshold: %f FRR: %f, FAR: %f" % (threshold, frr, far))






''' Test for ivector-PLDA-based OSI  
##### Note: the ways of setting threshold for OSI is a little different from SV 
##### although the resulted thresholds of these two ways do not vary too much according to our experiments.
##### Refer to one of our reference paper "Open-set speaker identification using adapted gaussian mixture models" for more details.
'''
group_id = "test-iv-OSI"
iv_osi_model = iv_OSI(group_id, iv_model_list)
spk_ids = np.array(iv_osi_model.spk_ids)

audio_list = []
target_label_list = []
spk_iter = os.listdir(test_dir)
for spk_id in spk_iter:

    target_label = np.argwhere(spk_ids == spk_id).flatten()[0]

    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        audio_list.append(audio)
        target_label_list.append(target_label)

spk_iter = os.listdir(illegal_dir)
illegal_audio_list = []
for spk_id in spk_iter:
    spk_dir = os.path.join(illegal_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        illegal_audio_list.append(audio)

score_target = []
score_untarget = []

_, target_scores = iv_osi_model.make_decisions(audio_list, debug=debug, n_jobs=n_jobs)
max_spk_index = np.argmax(target_scores, axis=1)
keep_utt_index = np.argwhere(max_spk_index == target_label_list).flatten()
keep_max_scores = np.max(target_scores[keep_utt_index], axis=1)
score_target += [score for score in keep_max_scores]

_, untarget_scores = iv_osi_model.make_decisions(illegal_audio_list, debug=debug, n_jobs=n_jobs)
max_scores = np.max(untarget_scores, axis=1)
score_untarget += [score for score in max_scores]

threshold, frr, far = set_threshold(score_target, score_untarget)
IER_cnt = np.intersect1d(np.argwhere(target_scores[:, max_spk_index] >= threshold).flatten(),
                         np.argwhere(max_spk_index != target_label_list).flatten()).size
IER = IER_cnt * 100 / len(audio_list)

print("----- Test of ivector-PLDA-based OSI, result ---> threshold: %f, FRR: %f, IER: %f, FAR: %f -----" % (threshold, frr, IER, far))






''' Test for gmm-ubm-based OSI
'''
group_id = "test-gmm-OSI"
gmm_osi_model = gmm_OSI(group_id, gmm_model_list, ubm)
spk_ids = np.array(gmm_osi_model.spk_ids)

audio_list = []
target_label_list = []
spk_iter = os.listdir(test_dir)
for spk_id in spk_iter:

    target_label = np.argwhere(spk_ids == spk_id).flatten()[0]

    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        audio_list.append(audio)
        target_label_list.append(target_label)

spk_iter = os.listdir(illegal_dir)
illegal_audio_list = []
for spk_id in spk_iter:
    spk_dir = os.path.join(illegal_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for i, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        _, audio = read(path)
        illegal_audio_list.append(audio)

score_target = []
score_untarget = []

_, target_scores = gmm_osi_model.make_decisions(audio_list, debug=debug, n_jobs=n_jobs)
max_spk_index = np.argmax(target_scores, axis=1)
keep_utt_index = np.argwhere(max_spk_index == target_label_list).flatten()
keep_max_scores = np.max(target_scores[keep_utt_index], axis=1)
score_target += [score for score in keep_max_scores]

_, untarget_scores = gmm_osi_model.make_decisions(illegal_audio_list, debug=debug, n_jobs=n_jobs)
max_scores = np.max(untarget_scores, axis=1)
score_untarget += [score for score in max_scores]

threshold, frr, far = set_threshold(score_target, score_untarget)
IER_cnt = np.intersect1d(np.argwhere(target_scores[:, max_spk_index] >= threshold).flatten(),
                         np.argwhere(max_spk_index != target_label_list).flatten()).size
IER = IER_cnt * 100 / len(audio_list)

print("----- Test of gmm-ubm-based OSI, result ---> threshold: %f, FRR: %f, IER: %f, FAR: %f -----" % (threshold, frr, IER, far))
