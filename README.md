# FAKEBOB
Source code for paper "Who is real Bob? Adversarial Attacks on Speaker Recognition Systems"

## Having Cleaned Code
1. attack on GMM-UBM-based open-set identification (OSI) system

## Future Cleaned Code
1. attack on GMM-UBM-based close-set identification (CSI) and speaker verification (SV) systems
2. attack on ivector-PLDA-based systems
3. transferability attack
4. defense evaluation

## Setup
###### Note that we have only tested our code on Ubuntu 16.04 system.
To reproduce our experiments, you should complete the following setups:
1. Install Kaldi Toolkit according to the instruction on [Kaldi website](https://github.com/kaldi-asr/kaldi).
2. Install python 3. We recommend using Anaconda and Python Virtual Enviroment.

## Directory Structure
1. 'data' folder: contanins the voice datas used for experiments.
(a) 'data/test-set-2': contains five speakers and 25 voices, used as starting voices for attacking CSI system.
(b) 'data/illegal-set-2-all-right': contains four imposters and 20 voices, used as starting voices for attacking ivector-PLDA-based SV and OSI systems.
(c) 'data/illegal-set-2-all-right-GMM-UBM': contains four imposters and 20 voices, used as starting voices for attacking GMM-UBM-based SV and OSI systems.
2. 'GMM-UBM' folder: contains all related code of attacking GMM-UBM-based speaker recognition systems.
(a) 'GMM-UBM/kaldiHelper.py', 'GMM-UBM/kaldiHelper.sh': python warpper for Kaldi.
(b) 'GMM-UBM/GMM-UBM-model': contains five speaker models and UBM models
(c) 'GMM-UBM/open-set-identification': related code of attacking GMM-UBM-based OSI system.

