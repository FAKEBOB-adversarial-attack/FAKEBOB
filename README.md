# FAKEBOB
Source code for paper "Who is real Bob? Adversarial Attacks on Speaker Recognition Systems".

Demonstration Website: [FAKEBOB Website](https://sites.google.com/view/fakebob/home "FAKEBOB Website")

Our paper has been accepted by 42nd IEEE Symposium on Security and Privacy (**IEEE S&P, Oakland**), 2021.

Paper link [Who is real Bob? Adversarial Attacks on Speaker Recognition Systems](https://arxiv.org/abs/1911.01840).

Cite our paper as follow:

    @inproceedings{chen2019real,
                   title={Who is Real Bob? Adversarial Attacks on Speaker Recognition Systems},
                   author={Chen, Guangke and Chen, Sen and Fan, Lingling and Du, Xiaoning and Zhao, Zhe and Song, Fu and Liu, Yang},
                   booktitle={Process dings of the 42nd IEEE Symposium on Security and Privacy (IEEE S&P, Oakland)},
                   year={2021}
        }

## Basic
#### Note that we have only tested our code on Ubuntu 16.04 system. It should work well as well on other Linux systems.

Reproduce our experiment step by step:
1. Install Python 3.  
    - Required packages: numpy, scipy, shlex, shutil.  
    - We recommend using Anaconda and Python Virtual Enviroment.
2. Download and install Kaldi Toolkit.
    - Download kaldi from [Kaldi website](https://github.com/kaldi-asr/kaldi "kaldi website").  
    Let ***$KALDI_ROOT$*** be the root path of your download. Root path means the parent directory of src/, egs/, etc.
    - Add the C file **gmm-global-est-map.cc** in our reposity to ***$KALDI_ROOT$/src/gmmbin/***.  
    This file will be complied to an executable file which is needed by GMM-UBM system.
    - Build the toolkit according to the instruction in ***$KALDI_ROOT$/INSTALL***.
3. After successfully building Kaldi toolkit, run the **egs/voxceleb/v1** recipe.  
    - download the VoxCeleb1 and VoxCeleb2 dataset from [VoxCeleb1 Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/ "VoxCeleb1"), [VoxCeleb2 Dataset](http://www.robots.ox.ac.uk/~vgg/data/voxceleb2/ "VoxCeleb2").
    - change directory to egs/voxceleb/v1.  
    - modify the variables **voxceleb1_root** and **voxceleb2_root** in line 19 and line 20 of **run.sh** to your actual location of datasets.  
    - run the shell scipt `./run.sh`. You may need to resolve permission denied problem by shell command `chmod 777 ./run.sh`
4. After **egs/voxceleb/v1** recipe completes running, copy (or make soft link for) the following files or directories to the *pre-models* directory of this reposity.  
    - change directory to your location of FAKEBOB and create a new directory named ***pre-models***.
    - copy (or make soft link) *final.dubm*, *final.ubm*, *final.ie*, *delta_opts* in ***$KALDI_ROOT$/egs/voxceleb/v1/exp/extractor/*** to ***pre-models/***.  
    - copy (or make soft link) *plda*, *mean.vec*, *transform.mat* in ***voxceleb/v1/exp/ivectors_train/*** to ***pre-models/***.  
    - copy (or make soft link) *conf/* in ***voxceleb/v1*** to ***pre-models/***.  
    - copy (or make soft link) *sid/*, *steps/*, *utils/* in ***voxceleb/v1*** to ***pre-models/***.

5. Downoad our dataset. Our used dataset comes from **LibriSpeech**. We just select part of it and convert the *.flac* format to *.wav* format.  
    - download data.tgz from [data.tgz, 920MB](https://drive.google.com/open?id=1oFCF3Ggw9xIyUDcYfmtUHd5ih1jUY79v "data.tgz").  
    - untar data.tgz to the location of FAKEBOB, after which you will see a new directory named *data/*.  
    Inside *data/*, there are four sub directories, i.e., *enrollment-set*, *z-norm-set*, *test-set* and *illegal-set*.
6. Setting the System Variable which ***kaldi*** relies on. 
We have made it as simple as possible. 
You just need to copy and paste all the commands in ***path_cmd.sh*** to your
***~/.zshrc*** or ***~/.bashrc***. 
Do not forget to modify **KALDI_ROOT** and **FAKEBOB_PATH** in ***path_cmd.sh***.
    - vim ~/.zshrc
    - modify **KALDI_ROOT** and **FAKEBOB_PATH** in ***path_cmd.sh***
    - copy and paste all the commands to ~/.zshrc
    - source ~/.zshrc
7. Build speaker unique models for enrolled speakers in *enrollment-set*.  
    - Running the python file `build_spk_models.py`.   
    - After running completed, you will see several new directories, among which ***models/*** stores the speaker unique models of ivector-PLDA system (in the form of ID.iv) and GMM-UBM system (in the form of ID.gmm).
8. Testing the baseline performance of ivector-PLDA-based and GMM-UBM-based OSI, CSI, SV systems.  
    - Running the python file `test.py`.  
    - During running, the baseline performance will be displayed in your terminal.  
As you can see, all of these systems are well-performed without attack.
9. Generate adversarial voices for speaker recognition systems (launch our attack FAKEBOB).
    - modify the variable KALDI_ROOT in ***attackMain.sh*** to your root path of kaldi.
    - adjust the parameters in ***attackMain.sh***.  
    For example, if you would like to launch targeted attack against ivector-PLDA-based CSI, just set `archi=iv, task=CSI,attack_type=targeted`.  
    You can also adjust other parameters which is associated with the efficiency and effectiveness of our attack FAKEBOB such as `epsilon`.
    - The generated adversarial voices are stored in **adversarial-audio**, and the corresponding checkpoints about the iterative procedures are in **checkpoint**.
    - The execuation time for each iteration depends on your machines.  
    On our machine, about 12s for ivector-PLDA and 5s for GMM-UBM.  
    You can adjust `n_jobs` and `samples_per_draw` to decrease this time.  
    But note that smaller `samples_per_draw` may lead to slower covergence and too large `n_jobs` may increase the time in contrast.

Since step 3 consumes quite a long time, you can choose to download ***pre-models.tgz*** from [pre-models.tgz, 493MB](https://drive.google.com/open?id=1T_hx9Pqopk-rlmiSrBWdXjl825wjBQVF "pre-models.tgz"). After downloading, just untar it to the location of FAKEBOB. Then you can skip both step 3 and step 4.

## Extension
If you would like to use your own dataset or attack other speaker recognition systems (e.g., DNN-based systems), here are some tips.

> ### Use your own dataset  
Each audio file should be in .wav format and named in the form of **ID-XXX.wav** (e.g., *1580-141084-0048.wav*) where ID denotes the unique speaker id of the speaker.  
You can exploit ffmpeg tool to change the format of your own audios.  
If your audios are not sampled at 16 KHZ or not quantized with 16 bits, you need to modify the `fs` and `bits_per_sample` in some .py file or specific them when calling some functions.  
>
Here are the specification of the four sub directories in *data/*.  It should help you prepare your own dataset more easily.  
1. ***enrollment-set***: voice datas used for enrollment. One audio corresponds to one enrolled speaker.  
2. ***z-norm-set***: voice data from imposters to obtain z_norm mean and z_norm std for scoring normalization.  
3. ***test-set***: each sub-folder contains voice datas of one enrolled speaker in enrollment-set.  
The name of each sub-folder should be the unique speaker ID of the corresponding speaker.
4. ***illegal-set***: each sub-folder contains voice datas of one imposter (imposters are not in enrollment-set).  
The name of each sub-folder should be the unique speaker ID of the corresponding speaker.
> ### Attack other speaker recognition systems  
To generate adversarial voices on other speaker recognition systems using our attack FAKEBOB, what you need to do is quite simple: just wrap your system by providing two interface - function `score` and `make_decisions`. Please refer to `gmm_ubm_OSI.py`, `gmm_ubm_CSI.py`, `gmm_ubm_SV.py`, `ivector_PLDA_OSI.py`, `ivector_PLDA_CSI.py`, `ivector_PLDA_SV.py` for details about the input and output arguments.

### If you have any question, feel free to comment or contact.


