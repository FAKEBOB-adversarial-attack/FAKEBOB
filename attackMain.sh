
# put your KALDI_ROOT here, KALDI_ROOT is the parent directory of egs, src, etc.
KALDI_ROOT=/root/gkchen/kaldi

########################################################
# This small part of code is directly drawn from KALDI_ROOT/tools/config/command_path.sh
# to add executable file to the PATH variable which is needed by ivector-PLDA and GMM-UBM systems 
# Copyright: KALDI developers

# we assume KALDI_ROOT is already defined
[ -z "$KALDI_ROOT" ] && echo >&2 "The variable KALDI_ROOT must be already defined" && exit 1
# The formatting of the path export command is intentionally weird, because
# this allows for easy diff'ing
export PATH=\
${KALDI_ROOT}/src/bin:\
${KALDI_ROOT}/src/chainbin:\
${KALDI_ROOT}/src/featbin:\
${KALDI_ROOT}/src/fgmmbin:\
${KALDI_ROOT}/src/fstbin:\
${KALDI_ROOT}/src/gmmbin:\
${KALDI_ROOT}/src/ivectorbin:\
${KALDI_ROOT}/src/kwsbin:\
${KALDI_ROOT}/src/latbin:\
${KALDI_ROOT}/src/lmbin:\
${KALDI_ROOT}/src/nnet2bin:\
${KALDI_ROOT}/src/nnet3bin:\
${KALDI_ROOT}/src/nnetbin:\
${KALDI_ROOT}/src/online2bin:\
${KALDI_ROOT}/src/onlinebin:\
${KALDI_ROOT}/src/rnnlmbin:\
${KALDI_ROOT}/src/sgmm2bin:\
${KALDI_ROOT}/src/sgmmbin:\
${KALDI_ROOT}/src/tfrnnlmbin:\
${KALDI_ROOT}/src/cudadecoderbin:\
$PATH
####################################################

spk_ids="1580 2830 4446 5142 61"

archi=gmm
# archi=iv
task=OSI
# task=CSI
# task=SV
attack_type=targeted
# attack_type=untargeted

adver_thresh=0.0
epsilon=0.002
max_iter=1000
max_lr=0.001
min_lr=1e-6
samples=50
sigma=0.001
momentum=0.9
plateau_length=5
plateau_drop=2.0

n_jobs=5
# debug=False

python attackMain.py -spk_id $spk_ids -archi $archi -task $task -type $attack_type \
-adver $adver_thresh -epsilon $epsilon -max_iter $max_iter -max_lr $max_lr \
-min_lr $min_lr -samples $samples -sigma $sigma -momentum $momentum \
-plateau_length $plateau_length -plateau_drop $plateau_drop \
-nj $n_jobs
