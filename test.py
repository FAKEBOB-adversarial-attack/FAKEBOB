
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
n_jobs = 30

test_dir = "./data/test-set"
illegal_dir = "./data/illegal-set"

model_dir = "model"
spk_id_list = ["1580", "2830", "4446", "5142", "61"] # Change to your own spk ids !!!!
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
print("----- Test of ivector-PLDA-based CSI, result ---> %d/%d, Accuracy:%f ----- " % (correct_cnt, decisions.size, acc)),





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
print("----- Test of gmm-ubm-based CSI, result ---> %d/%d, Accuracy:%f ----- " % (correct_cnt, decisions.size, acc)),





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

frr_cnt = 0
target_cnt = 0
far_cnt = 0
untarget_cnt = 0
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

    target_cnt += len(audio_list)
    
    decisions, _ = iv_sv_model.make_decisions(audio_list, n_jobs=n_jobs, debug=debug)
    decisions = np.array(decisions)
    frr_cnt += np.argwhere(decisions == -1).flatten().size

    decisions, _ = iv_sv_model.make_decisions(illegal_audio_list, n_jobs=n_jobs, debug=debug)
    decisions = np.array(decisions)
    far_cnt += np.argwhere(decisions == 1).flatten().size

    untarget_cnt += len(illegal_audio_list)

print("----- Test of ivector-PLDA-based SV, result ---> FRR: %d/%d, %f, FAR: %d/%d, %f" %(frr_cnt, target_cnt, 
      frr_cnt * 100 / target_cnt, far_cnt, untarget_cnt, far_cnt * 100 / untarget_cnt))





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

frr_cnt = 0
target_cnt = 0
far_cnt = 0
untarget_cnt = 0
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

    target_cnt += len(audio_list)
    
    decisions, _ = gmm_sv_model.make_decisions(audio_list, n_jobs=n_jobs, debug=debug)
    decisions = np.array(decisions)
    frr_cnt += np.argwhere(decisions == -1).flatten().size

    decisions, _ = gmm_sv_model.make_decisions(illegal_audio_list, n_jobs=n_jobs, debug=debug)
    decisions = np.array(decisions)
    far_cnt += np.argwhere(decisions == 1).flatten().size

    untarget_cnt += len(illegal_audio_list)

print("----- Test of gmm-ubm-based SV, result ---> FRR: %d/%d, %f, FAR: %d/%d, %f" %(frr_cnt, target_cnt, 
      frr_cnt * 100 / target_cnt, far_cnt, untarget_cnt, far_cnt * 100 / untarget_cnt))





''' Test for ivector-PLDA-based OSI
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

frr_cnt = 0
target_cnt = len(audio_list)
far_cnt = 0
untarget_cnt = len(illegal_audio_list)

decisions, _ = iv_osi_model.make_decisions(audio_list, debug=debug, n_jobs=n_jobs)
decisions = np.array(decisions)
target_label_list = np.array(target_label_list)
frr_cnt = np.argwhere(decisions == -1).flatten().size
ier_cnt = np.intersect1d(np.argwhere(decisions == 1).flatten(), np.argwhere(decisions != target_label_list)).size

decisions, _ = iv_osi_model.make_decisions(illegal_audio_list, debug=debug, n_jobs=n_jobs)
decisions = np.array(decisions)
far_cnt = np.argwhere(decisions == 1).flatten().size

print("----- Test of ivector-PLDA-based OSI, result ---> FRR: %d/%d, %f, IER: %d/%d, %f, FAR: %d/%d, %f -----" %(frr_cnt, target_cnt, 
     frr_cnt * 100 / target_cnt, ier_cnt, target_cnt, ier_cnt * 100 / target_cnt, far_cnt, untarget_cnt, far_cnt * 100 /untarget_cnt))





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

frr_cnt = 0
target_cnt = len(audio_list)
far_cnt = 0
untarget_cnt = len(illegal_audio_list)

decisions, _ = gmm_osi_model.make_decisions(audio_list, debug=debug, n_jobs=n_jobs)
decisions = np.array(decisions)
target_label_list = np.array(target_label_list)
frr_cnt = np.argwhere(decisions == -1).flatten().size
ier_cnt = np.intersect1d(np.argwhere(decisions == 1).flatten(), np.argwhere(decisions != target_label_list)).size

decisions, _ = gmm_osi_model.make_decisions(illegal_audio_list, debug=debug, n_jobs=n_jobs)
decisions = np.array(decisions)
far_cnt = np.argwhere(decisions == 1).flatten().size

print("----- Test of gmm-ubm-based OSI, result ---> FRR: %d/%d, %f, IER: %d/%d, %f, FAR: %d/%d, %f -----" %(frr_cnt, target_cnt, 
     frr_cnt * 100 / target_cnt, ier_cnt, target_cnt, ier_cnt * 100 / target_cnt, far_cnt, untarget_cnt, far_cnt * 100 /untarget_cnt))