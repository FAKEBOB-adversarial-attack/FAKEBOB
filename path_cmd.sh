# copy and paste the following command to your ~/.zshrc or ~/.bashrc
# do not forget to modify KALDI_ROOT and FAKEBOB_PATH

export KALDI_ROOT=/root/gkchen/kaldi # change to your own KAIDI_ROOT
FAKEBOB_PATH=/root/gkchen/FAKEBOB # change to your FAKEBOB PATH

export PATH=$FAKEBOB_PATH/pre-models/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
export train_cmd="run.pl --mem 4G" # run.pl for running kaldi local machine