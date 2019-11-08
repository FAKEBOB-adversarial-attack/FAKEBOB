
'''
Note: Some of the codes in this .py file are inspired by
https://github.com/labsix/limitedblackbox-attacks
'''

import numpy as np
import copy
import time
import pickle

class blackBoxAttack(object):

    def __init__(self, model, max_iter=1000, epsilon=0.0002, 
                min_lr=1e-6, max_lr=1e-3, samples_per_draw=50, 
                batchsize=50, n_jobs=1, sigma=1e-3, momentum=0.9,
                plateau_length=5, plateau_drop=2.0, adver_thresh=0.
                ):

        self.model = model
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.samples_per_draw = samples_per_draw
        self.batchsize = batchsize
        self.n_jobs = n_jobs
        self.sigma = sigma
        self.momentum = momentum
        self.plateau_length = plateau_length
        self.plateau_drop = plateau_drop

        self.adver_thresh = adver_thresh

        if not self.samples_per_draw % self.batchsize == 0:
            print("--- Warning:samples_per_draw must be multiple interger of batchsize ---")
        
        if not self.batchsize % self.n_jobs == 0:
            print("--- Warning:batchsize must be multiple interger of n_jobs ---")

    def attack(self, audio, threshold, checkpoint_path, target, fs=16000, 
               bits_per_sample=16, n_jobs=28, debug=False
              ):
        
        # make sure that audio is (N, 1)
        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] == 1:
            audio = audio.T
        else:
            pass

        self.threshold = threshold
        """ initial
        """
        adver = copy.deepcopy(audio)
        pre_adv = copy.deepcopy(adver)
        grad = 0
        pre_grad = grad

        last_ls = []

        max_lr = self.max_lr

        lower = np.clip(audio - self.epsilon, -1., 1.)
        upper = np.clip(audio + self.epsilon, -1., 1.)

        cp_global = []

        self.target = target

        for iter in range(self.max_iter):

            start = time.time()

            cp_local = []

            # first check should whether stop early
            flag, score, single_loss = self.whether_adver(adver, fs=fs)
            distance = np.max(np.abs(audio - adver))
            print("--- iter %d, distance:%f, loss:%f, score: ---"%(iter, distance, single_loss), score)
            if flag == 1:
                print("------ early stop at iter %d ---"%iter)

                cp_local.append(distance)
                cp_local.append(single_loss)
                cp_local.append(score)
                cp_local.append(0.)

                cp_global.append(cp_local)

                break
            
            # estimate the grad
            pre_grad = copy.deepcopy(grad) 
            loss, grad = self.get_grad(adver, fs=fs, n_jobs=n_jobs, debug=debug)
            grad = self.momentum * pre_grad + (1.0 - self.momentum) * grad

            last_ls.append(loss)
            last_ls = last_ls[-self.plateau_length:]
            if last_ls[-1] > last_ls[0] and len(last_ls) == self.plateau_length:
                if max_lr > self.min_lr:
                    print("[log] Annealing max_lr")
                    max_lr = max(max_lr / self.plateau_drop, self.min_lr)
                last_ls = []
            
            pre_adv = copy.deepcopy(adver)
            adver -= max_lr * np.sign(grad)
            #adver = np.minimum(1.0, audio + self.epsilon, np.maximum(-1.0, audio - self.epsilon, adver))
            adver = np.clip(adver, lower, upper)

            end = time.time()
            used_time = end -start
            print("consumption time:%f, lr:%f"%(used_time, max_lr))
            
            cp_local.append(distance)
            cp_local.append(single_loss)
            cp_local.append(score)
            cp_local.append(used_time)

            cp_global.append(cp_local)
        
        with open(checkpoint_path, "wb") as writer:
            pickle.dump(cp_global, writer, protocol=-1)
        
        success_flag = 1 if iter < self.max_iter-1 else -1
        adver = (adver * (2 ** (bits_per_sample - 1))).astype(np.int16)
        return adver, success_flag

    def whether_adver(self, audio, fs=16000):

        n_jobs = 1
        debug = False

        score = self.model.score(audio, fs=fs, n_jobs=n_jobs, debug=debug) # score is (n_speakers, )
        score_other = np.delete(score, self.target)
        loss = np.maximum(np.maximum(np.max(score_other), self.threshold) - score[self.target], -1 * self.adver_thresh)
        #loss = np.maximum(self.threshold - score, -1 * self.adver_thresh) # (batches_per_job, 1)

        flag = -1

        if loss == -1 * self.adver_thresh:
            flag = 1

        return flag, score, loss
    
    def get_grad(self, audio, fs=16000 ,n_jobs=28, debug=False):

        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] == 1:
            audio = audio.T
        else:
            pass
        
        N = audio.size

        n_batches = self.samples_per_draw // self.batchsize

        grad_batch = np.zeros((N, n_batches), dtype=np.float64)
        loss_batch = np.zeros(n_batches, dtype=np.float64)

        for i in range(n_batches):

            grad_job = np.zeros((N, self.n_jobs), dtype=np.float64)
            batches_per_job = self.batchsize // self.n_jobs
            loss_job = np.zeros((batches_per_job, self.n_jobs), dtype=np.float64)

            for j in range(self.n_jobs):

                noise_pos = np.random.normal(size=(N, batches_per_job // 2))
                noise = np.concatenate((noise_pos, -1. * noise_pos), axis=1)
                noise_audios = self.sigma * noise + audio
                loss = self.loss_fn(noise_audios, fs=fs, n_jobs=n_jobs, debug=debug) # loss is (batches_per_job, 1)
                loss_job[:, j:j+1] = loss
                grad_job[:, j:j+1] = np.mean(loss.flatten() * noise, axis=1, keepdims=True) / self.sigma # grad is (N,1)
            
            grad_batch[:, i:i+1] = np.mean(grad_job, axis=1, keepdims=True)
            loss_batch[i] = np.mean(loss_job)
        
        estimate_grad = np.mean(grad_batch, axis=1, keepdims=True)
        final_loss = np.mean(loss_batch)
    
        return final_loss, estimate_grad # scalar, (N,1)
    
    def loss_fn(self, audios, fs=16000, n_jobs=28, debug=False):

        #n_jobs = 3 * audios.shape[1]
        #n_jobs = 28
        #debug = False

        score = self.model.score(audios, fs=fs, n_jobs=n_jobs, debug=debug) # score is (batches_per_job, n_speakers)
        score_other = np.delete(score, self.target, axis=1) #score_other is (batches_per_job, n_speakers-1)
        score_real = np.max(score_other, axis=1)[:, np.newaxis] # score_real is (batches_per_job, 1)
        score_target = score[:, self.target:self.target+1] # score_target is (batches_per_job, 1)
        loss = np.maximum(np.maximum(score_real, self.threshold) - score_target, -1 * self.adver_thresh)
        return loss # loss is (batches_per_job, 1)













        

        


