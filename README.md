# FAKEBOB
Source code for paper "Who is real Bob? Adversarial Attacks on Speaker Recognition Systems".

Paper link: [PrePrint Version](https://arxiv.org/abs/1911.01840)

Demonstration Website: [FAKEBOB Website](https://sites.google.com/view/fakebob/home)

## Setup
#### Note that we have only tested our code on Ubuntu 16.04 system. It should work well as well on other Linux systems.

To reproduce our experiments, you should complete the following setups:
1. Install Python 3. Required packages: numpy, scipy, shlex, shutil.
We recommend using Anaconda and Python Virtual Enviroment.
2. Install Kaldi Toolkit according to the instruction on [Kaldi website](https://github.com/kaldi-asr/kaldi).
Tips drawn from Kaldi website: To build the toolkit: see the instructions in $KALDI_ROOT$/INSTALL.
**Before Building, please add the C file gmm-global-est-map.cc to the directory $KALDI_ROOT$/src/gmmbin**. This file will be complied to an executable file which is needed by GMM-UBM system.
3. After successfully building Kaldi toolkit, run the 'voxceleb/v1' recipe: change directory to egs/voxceleb/v1 and run the shell scipt ./run.sh. Tips: Beforing runing this shell script, you should download the VoxCeleb1 and VoxCeleb2 dataset from [VoxCeleb1 Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/), [VoxCeleb2 Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/) and modify the variables 'voxceleb1_root' and 'voxceleb2_root' in line 19 and line 20 of 'run.sh' to your actual location of datasets.
4. After 'voxceleb/v1' recipe completes running, copy (or make soft link) the following files or directories to the 'pre-models' directory of this reposity:
(a) final.dubm, final.ubm, final.ie, delta_opts in 'voxceleb/v1/exp/extractor'
(b) plda, mean.vec, transform.mat in 'voxceleb/v1/exp/ivectors_train'
(c) conf/ in voxceleb/v1
(d) sid/, steps/, utils/ in voxceleb/v1
5. Prepare your own voice datas. If you choose to use our defult voice datas, you can just skip this step.
(a) enrollment-set: voice datas used for enrollment. One audio corresponds to one enrolled speaker.
(b) z-norm-set: voice data from imposters to obtain z_norm mean and z_norm std for scoring normalization.
(c) test-set: each sub-folder contains voice datas of one enrolled speaker in enrollment-set.
(d) illegal-set: each sub-folder contains voice datas of one imposter (imposters are not in enrollment-set).
**Note: each audio file should in .wav format and named in the form of "ID-XXX.wav" where ID denotes the unique speaker id of the speaker. You can exploit ffmpeg tool to change the format of your own audios**
6. Our provided voice datas are from LibriSpeech (LibriSpeech provides .flac format, we convert them into .wav format). To use out dataset, just untar the data.tgz. **Note that because Github has file size limit, we can only upload a small part of our used datas.**

## Tips
1. run bulid_sok_models.py to generate speaker unique models for enrolled speaker in enrollment-set which are stored in model/.
2. run test.py to test the baseline performance of ivector-PLDA-based and GMM-UBM-based OSI, CSI, SV systems.
3. run attackMain.sh to launch our attack FAKEBOB (generate adversarial voices). You can easily modify the parameters of our attack FAKEBOB in attackMain.sh. **Note: do not forget to modify the variable KALDI_ROOT in attackMain.sh to your root path of kaldi (the root path means the parent dir of egs, src, etc).**
4. The iteration time depends on your machines. On our machine, about 12s for ivector-PLDA and 5s for GMM-UBM. You can adjust n_jobs and samples_per_draw to decrease the iteration time. But note that smaller samples_per_draw can leads to slower covergence and too great n_jobs may lower the speed in contrast.

### If you have any question, feel free to comment or contact.


