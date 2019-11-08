
import os
import numpy as np
from scipy.io.wavfile import read
from scipy.io.wavfile import write
from blackBoxAttack_NES import blackBoxAttack
from openSetIdentificationModel import openSetIdentificationModel
import random

bitsPerSample = 16
fs = 16000
    
def loadData(model, spk_id):

    dataPath = "../../data/illegal-set-2-all-right-GMM-UBM"

    data = []
    target = []

    target_value = np.argwhere(np.array(model.spk_id_list) == spk_id)[0][0]

    spk_iter = os.listdir(dataPath)
    for spk_id in spk_iter:
        spk_dir = os.path.join(dataPath, spk_id)
        audioIter = os.listdir(spk_dir)
    #print('target_value', target_value)
    
        for j in range(len(audioIter)):
        #for j in range(5):

            audio = os.path.join(spk_dir, audioIter[j])
            fs, signal = read(audio)
            signal = signal / (2 ** (bitsPerSample - 1))
        
            #decision, _=model.make_decision(signal[:, np.newaxis], fs)
            #if decision==-1:
            if True:
                data.append(signal)
                target.append(target_value)
                audio_name.append(audioIter[j])
                save_audio_name.append(os.path.join(save_path, audioIter[j]))
                checkpoint_name.append(os.path.join(checkpoint_path, os.path.splitext(audioIter[j])[0]))
        
    return data, target # return list (list element is np.array with the shape (len, )) instead of array

def load_model(group_id, spk_id_list, model_dir, threshold):

    spk_model_list = [os.path.join(model_dir, spk_id + ".model") for spk_id in spk_id_list]
    #model_file = os.path.join(model_dir, spk_id + ".model")
    ubm = os.path.join(model_dir, "final.dubm")
    model = openSetIdentificationModel(group_id, spk_id_list, spk_model_list, ubm, threshold=threshold)
    
    return model

def estimate_threshold(attack, allInput, allTarget, estimate_num):
    
    #estimate_num=int(total_num*estimate_ratio/100)
    
    #all_index=list(range(len(allInput)))
    '''
    all_index=np.arange(len(allInput))
    selected_index=np.random.sample(all_index, estimate_num)
    estimate_input=allInput[selected_index]
    estimate_target=allTarget[selected_index]
    '''
    estimate_input = allInput[:estimate_num]
    estimate_target = allTarget[:estimate_num]
    #estimate_input=allInput[0]
    
    
    threshold_array = np.zeros(estimate_num, dtype=np.float64)
    
    iter_cost=0
    
    for k, audio in enumerate(estimate_input):
        
        print('------ %dth audio ------'%k)
        threshold_array[k], cost=attack.estimate_threshold(audio[:, np.newaxis], estimate_target[k])
        print('---- %dth auudio, estimated threshold:%f ----'%(k, threshold_array[k]))
        
        iter_cost=iter_cost+cost
    
    threshold_array=np.delete(threshold_array, np.argwhere(threshold_array==np.infty))
    threshold_mean=np.mean(threshold_array)
    threshold_std=np.std(threshold_array)
    threshold=threshold_mean+threshold_std
    
    return threshold, iter_cost

def generate_adversarial_audio(allInput, allTarget, threshold, audio_name, checkpoint_name, 
                               save_audio_name, attack, model, nj, debug, start, end):
    
    total_cnt = len(allInput)
    success_cnt = 0
    
    iter_cost = 0
    
    #for i in range(total_cnt):
    for i in range(start, end):
    #for i in range(1):
        audio = allInput[i][:, np.newaxis] # audio is (len, 1)
        target = allTarget[i]
        print('------ audio_name[%d]: %s, target:%d ------'%(i, save_audio_name[i], target))
        #decision, score=model.make_decision(audio, fs)
        #print('------ score of original audio ------', score)
        #print('------ decision of original audio ------', decision)
        naudio, success_flag = attack.attack(audio, threshold, checkpoint_name[i], target, n_jobs=nj, debug=debug)
        ## eval code, temporary only consider whether target, don't consider l2dist
        #naudio=naudio*(2**(bitsPerSample))
        #naudio=naudio.astype(np.int16)
        #decision, score=model.make_decision(naudio, fs)
        #print('------ score of new audio ------', score)
        #print('------ decision of new audio ------', decision)
        #if decision==target:
            #success_cnt+=1
            #print('------generate adversarialAudio SUCCESSFUL!!!------ ')

        if success_flag:
            success_cnt += 1

        write(save_audio_name[i], fs, naudio)
            
        #iter_cost=iter_cost+cost
            
        #else:
            #print('------generate adversarialAudio FAILED!!!------ ')
    
    #return success_cnt*100/total_cnt
    return total_cnt, success_cnt, iter_cost

#def main(id, estimate_ratio):
def main(spk_id, group_id, spk_id_list, model_dir, threshold_model, 
        epsilon, max_iter, min_lr, max_lr, samples_per_draw, batchsize, n_jobs, sigma, 
        momentum, plateau_length, plateau_drop, adver_thresh, 
        threshold_estimated, nj, debug, start, end):
    
    global audio_name
    audio_name = []
    global save_audio_name
    save_audio_name = []
    global checkpoint_name
    checkpoint_name = []

    global save_path
    save_path = os.path.join('adversarial-audio', group_id)
    global checkpoint_path
    checkpoint_path = os.path.join('checkpoint', group_id)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    '''
    load model
    '''
    print('------ load model ------')
    model = load_model(group_id, spk_id_list, model_dir, threshold_model)
    print('------ load model done ------')
    
    '''
    load data
    '''
    print('------ load data ------')
    allInput, allTarget = loadData(model, spk_id)
    total_num = len(allInput)
    print('------ load data done, total num: %d ------'%total_num)
    
    attack = blackBoxAttack(model, max_iter=max_iter, epsilon=epsilon, 
        min_lr=min_lr, max_lr=max_lr, samples_per_draw=samples_per_draw, 
        batchsize=batchsize, n_jobs=n_jobs, sigma=sigma, momentum=momentum, 
        plateau_length=plateau_length, plateau_drop=plateau_drop, adver_thresh=adver_thresh)
    
    '''
    estimate the threshold of the system
    '''
    #print('------ estimate threshold of speaker %s ------'%id)
    #threshold, cost=estimate_threshold(attack, allInput, allTarget, estimate_num)
    #print('------ estimate threshold of speaker %s done, estimated threshold: %f iter_cost %d ------'%(id, threshold, cost))

    threshold = threshold_estimated
    
    '''
    generate adversarialAudio
    '''
    print('------ generate adversarialAudio for speaker %s ------'%spk_id)
    total_cnt, success_cnt, iter_cost = generate_adversarial_audio(allInput, allTarget, threshold,
                                                                   audio_name, checkpoint_name,
                                                                   save_audio_name, attack, model,
                                                                   nj, debug, start, end)
    #print('------ total_cnt %d, success_cnt %d ------ '%(total_cnt, success_cnt))
    print('------ attack successful rate %d ------'%(success_cnt * 100 / total_cnt))
    #print('------ total_iter_cost %d ------'%iter_cost)
    #print('------ generate adversarialAudio for speaker %s done ------'%id)
    
    return

if __name__ == "__main__":
    
    epsilon = 0.002

    max_iter = 1000
    min_lr = 1e-6
    max_lr = 0.001

    samples_per_draw = 50
    batchsize = 50
    n_jobs = 1

    sigma = 1e-3
    momentum = 0.9
    plateau_length = 5
    plateau_drop = 2.0

    adver_thresh = 0.

    threshold_model = 0.0913
    threshold_estimated = 0.092

    model_file = "../model/speaker_model"

    nj = 5
    debug = False

    start = 0
    end = 20

    spk_id_list = ["1580", "2830", "4446", "5142", "61"]
    #spk_id_list = ["1580", "2830", "4446", "5142"]
    model_dir = "../GMM-UBM-model"
    for spk_id in spk_id_list:
        group_id = spk_id + "-" + str(adver_thresh)
        main(spk_id, group_id, spk_id_list, model_dir, threshold_model, 
            epsilon, max_iter, min_lr, max_lr, samples_per_draw, batchsize, n_jobs, sigma, 
            momentum, plateau_length, plateau_drop, adver_thresh, 
            threshold_estimated, nj, debug, start, end)
    
