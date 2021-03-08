#!/bin/bash

set -x 
command_to_execute="$@"

if [[ $(basename $(realpath .)) == "docker" ]]
then
    cd ..
fi

if [[ $command_to_execute == "" ]]
then
    command_to_execute="bash"
fi

docker run -it --gpus all -v $PWD:/mounted fakebob bash -ic ' \
export KALDI_ROOT=/kaldi; \
FAKEBOB_PATH=/mounted; \
export PATH=$FAKEBOB_PATH/pre-models/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5:$PATH; \
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1; \
. $KALDI_ROOT/tools/config/common_path.sh; \
export LC_ALL=C; \
export train_cmd="run.pl --mem 4G"; \
cd /mounted; \
'"$command_to_execute"
