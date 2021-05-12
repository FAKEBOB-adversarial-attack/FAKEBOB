
## Copyright (C) 2019, Guangke Chen <gkchen.shanghaitech@gmail.com>.
## This program is licenced under the BSD 2-Clause licence
## contained in the LICENCE file in this directory.


# Note: Some of the codes in this .py file are inspired by
# https://github.com/labsix/limitedblackbox-attacks


import copy
import pickle
import time

import numpy as np

UNTARGETED = "untargeted"

class FakeBob(object):

    def __init__(self, task, attack_type, model, adver_thresh=0., epsilon=0.002, max_iter=1000, 
                 max_lr=0.001, min_lr=1e-6, samples_per_draw=50, sigma=0.001, momentum=0.9, 
                 plateau_length=5, plateau_drop=2.):

        self.task = task
        self.attack_type = attack_type
        self.model = model
        self.adver_thresh = adver_thresh
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.samples_per_draw = samples_per_draw
        self.sigma = sigma
        self.momentum = momentum
        self.plateau_length = plateau_length
        self.plateau_drop = plateau_drop
    
    def estimate_threshold(self, audio, fs=16000, bits_per_sample=16, n_jobs=10, debug=False):

        if self.task == "CSI":
            print("--- Warning: no need to estimate threshold for CSI, quitting ---")
            return

        # make sure that audio is (N, 1)
        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] == 1:
            audio = audio.T
        else:
            pass

        init_score = self.model.score(audio, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
        if self.task == "OSI":
            init_score = np.max(init_score)
        
        self.delta = np.abs(init_score / 10)

        self.threshold = init_score + self.delta

        adver = copy.deepcopy(audio)
        grad = 0

        lower = np.clip(audio - self.epsilon, -1., 1.)
        upper = np.clip(audio + self.epsilon, -1., 1.)

        iter_outer = 0

        n_iters = 0
        times = 0

        # to estimate threshold, untargeted attack is enough
        attack_type_backup = self.attack_type
        self.attack_type = UNTARGETED

        while True:

            print("----- iter_outer:%d, threshold:%f -----" %(iter_outer, self.threshold))

            iter_inner = 0

            lr = self.max_lr # reset max_lr, or the iterative procedure wil be too slow
            last_ls = []

            while True:

                start = time.time()

                decision, score = self.model.make_decisions(adver, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                #distance = np.max(np.abs(audio - adver))
                print("--- iter_inner:%d, dicision:%d, score: ---" %(iter_inner, decision), score)

                if self.task == "OSI":
                        score = np.max(score)
                
                if decision != -1:
                    
                    print("--- return at iter_outer:%d, iter_inner:%d, return thresh:%f ---" %(iter_outer, iter_inner, score)),
                    print("cost %d iters, %fs time" %(n_iters, times)),

                    self.attack_type = attack_type_backup # change back

                    return score, n_iters, times
                
                elif score >= self.threshold:

                    print("--- early stop at iter_inner:%d ---" %(iter_inner))

                    break
                
                # estimate the grad
                pre_grad = copy.deepcopy(grad)
                loss, grad, _, _ = self.get_grad(adver, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)
                grad = self.momentum * pre_grad + (1.0 - self.momentum) * grad

                last_ls.append(loss)
                last_ls = last_ls[-self.plateau_length:]
                if last_ls[-1] > last_ls[0] and len(last_ls) == self.plateau_length:
                    if lr > self.min_lr:
                        lr = max(lr / self.plateau_drop, self.min_lr)
                    last_ls = []
                
                adver -= lr * np.sign(grad)
                adver = np.clip(adver, lower, upper)

                end = time.time()
                used_time = end -start
                print("consumption time:%f, lr:%f" %(used_time, lr))

                n_iters += 1
                times += used_time

                iter_inner += 1
            
            self.threshold += self.delta

            iter_outer += 1
    
    def attack(self, audio, checkpoint_path, threshold=0., true=None, target=None, fs=16000, 
               bits_per_sample=16, n_jobs=10, debug=False):
        
        # make sure that audio is (N, 1)
        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] == 1:
            audio = audio.T
        else:
            pass

        self.threshold = threshold
        self.true = true
        self.target = target

        """ initial
        """
        adver = copy.deepcopy(audio)
        grad = 0

        last_ls = []

        lr = self.max_lr

        lower = np.clip(audio - self.epsilon, -1., 1.)
        upper = np.clip(audio + self.epsilon, -1., 1.)

        cp_global = []

        for iter in range(self.max_iter):

            start = time.time()

            cp_local = []
            
            # estimate the grad
            pre_grad = copy.deepcopy(grad) 
            loss, grad, adver_loss, score = self.get_grad(adver, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)

            distance = np.max(np.abs(audio - adver))
            print("--- iter %d, distance:%f, loss:%f, score: ---" % (iter, distance, adver_loss), score)
            # if adver_loss == -1 * self.adver_thresh: # early stop condition for loss function with outermost maximum operation
            if adver_loss < 0: # early stop condition for loss function without outermost maximum operation
                print("------ early stop at iter %d ---" % iter)

                cp_local.append(distance)
                cp_local.append(adver_loss)
                cp_local.append(score)
                cp_local.append(0.)

                cp_global.append(cp_local)

                break

            grad = self.momentum * pre_grad + (1.0 - self.momentum) * grad

            last_ls.append(loss)
            last_ls = last_ls[-self.plateau_length:]
            if last_ls[-1] > last_ls[0] and len(last_ls) == self.plateau_length:
                if lr > self.min_lr:
                    lr = max(lr / self.plateau_drop, self.min_lr)
                last_ls = []
            
            adver -= lr * np.sign(grad)
            adver = np.clip(adver, lower, upper)

            end = time.time()
            used_time = end -start
            print("consumption time:%f, lr:%f"%(used_time, lr))
            
            cp_local.append(distance)
            cp_local.append(adver_loss)
            cp_local.append(score)
            cp_local.append(used_time)

            cp_global.append(cp_local)
        
        with open(checkpoint_path, "wb") as writer:
            pickle.dump(cp_global, writer, protocol=-1)
        
        success_flag = 1 if iter < self.max_iter-1 else -1
        adver = (adver * (2 ** (bits_per_sample - 1))).astype(np.int16)
        return adver, success_flag
    
    def get_grad(self, audio, fs=16000, bits_per_sample=16, n_jobs=10, debug=False):

        if len(audio.shape) == 1:
            audio = audio[:, np.newaxis]
        elif audio.shape[0] == 1:
            audio = audio.T
        else:
            pass
        
        N = audio.size

        noise_pos = np.random.normal(size=(N, self.samples_per_draw // 2))
        noise = np.concatenate((noise_pos, -1. * noise_pos), axis=1)
        noise = np.concatenate((np.zeros((N, 1)), noise), axis=1)
        noise_audios = self.sigma * noise + audio
        loss, scores = self.loss_fn(noise_audios, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug) # loss is (samples_per_draw + 1, 1)
        adver_loss = loss[0]
        score = scores[0]
        loss = loss[1:, :]
        noise = noise[:, 1:]
        final_loss = np.mean(loss)
        estimate_grad = np.mean(loss.flatten() * noise, axis=1, keepdims=True) / self.sigma # grad is (N,1)
    
        return final_loss, estimate_grad, adver_loss, score # scalar, (N,1)
    
    def loss_fn(self, audios, fs=16000, bits_per_sample=16, n_jobs=10, debug=False):

        score = self.model.score(audios, fs=fs, bits_per_sample=bits_per_sample, n_jobs=n_jobs, debug=debug)

        if self.task == "OSI": # score is (samples_per_draw + 1, n_spks)

            if self.attack_type == "targeted":

                score_other = np.delete(score, self.target, axis=1) #score_other is (samples_per_draw + 1, n_speakers-1)
                score_other_max = np.max(score_other, axis=1, keepdims=True) # score_real is (samples_per_draw + 1, 1)
                score_target = score[:, self.target:self.target+1] # score_target is (samples_per_draw + 1, 1)
                # the outermost maximum operation in the loss function will cause unexpected issue for benign voices whose initial loss is slightly larger than adver_thresh.
                # see README for detail
                # loss = np.maximum(np.maximum(score_other_max, self.threshold) - score_target, -1 * self.adver_thresh)
                loss = np.maximum(score_other_max, self.threshold) + self.adver_thresh - score_target

            else: 
                score_max = np.max(score, axis=1, keepdims=True) # (samples_per_draw + 1, 1)
                # the outermost maximum operation in the loss function will cause unexpected issue for benign voices whose initial loss is slightly larger than adver_thresh.
                # see README for detail
                # loss = np.maximum(self.threshold - score_max, -1 * self.adver_thresh)
                loss = self.threshold + self.adver_thresh - score_max
        
        elif self.task == "CSI": # score is (samples_per_draw + 1, n_spks)

            if self.attack_type == "targeted":

                score_other = np.delete(score, self.target, axis=1) #score_other is (samples_per_draw + 1, n_speakers-1)
                score_other_max = np.max(score_other, axis=1, keepdims=True) # score_real is (samples_per_draw + 1, 1)
                score_target = score[:, self.target:self.target+1] # score_target is (samples_per_draw + 1, 1)
                # the outermost maximum operation in the loss function will cause unexpected issue for benign voices whose initial loss is slightly larger than adver_thresh.
                # see README for detail
                # loss = np.maximum(score_other_max - score_target, -1 * self.adver_thresh)
                loss = score_other_max + self.adver_thresh - score_target
            
            else:

                score_other = np.delete(score, self.true, axis=1) #score_other is (samples_per_draw + 1, n_speakers-1)
                score_other_max = np.max(score_other, axis=1, keepdims=True) # score_real is (samples_per_draw + 1, 1)
                score_true = score[:, self.true:self.true+1] # score_target is (samples_per_draw + 1, 1)
                # the outermost maximum operation in the loss function will cause unexpected issue for benign voices whose initial loss is slightly larger than adver_thresh.
                # see README for detail
                # loss = np.maximum(score_true - score_other_max, -1 * self.adver_thresh)
                loss = score_true + self.adver_thresh - score_other_max
        
        else: # score is (samples_per_draw + 1, )
            # the outermost maximum operation in the loss function will cause unexpected issue for benign voices whose initial loss is slightly larger than adver_thresh.
            # see README for detail
            # loss = np.maximum(self.threshold - score[:, np.newaxis], -1 * self.adver_thresh)
            loss = self.threshold + self.adver_thresh - score[:, np.newaxis]

        return loss, score # loss is (samples_per_draw + 1, 1)
