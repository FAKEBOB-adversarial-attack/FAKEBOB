from kaldiasr/kaldi

run apt-get update && apt-get install -y  \
python3-pip \
git

run python3 --version

run pip3 install \
numpy==1.15

run pip3 install \
scipy==1.4.0

run pip3 install \
ushlex

run pip3 install \
pytest-shutil

run git clone https://github.com/FAKEBOB-adversarial-attack/FAKEBOB.git
env KALDI_ROOT="/opt/kaldi"
run cp FAKEBOB/gmm-global-est-map.cc $KALDI_ROOT/src/gmmbin/


run mkdir -p /export/corpora
run ln -s /mounted/data/vox1 /export/corpora/VoxCeleb1
run ln -s /mounted/data/vox2 /export/corpora/VoxCeleb2

workdir $KALDI_ROOT
run apt-get update && apt-get install -y  \
libopenblas-dev \
libatlas-base-dev \
cmake

workdir $KALDI_ROOT/tools
run apt-get update && apt-get install -y  \
oss-compat \
build-essential \
libdb1-compat
run extras/check_dependencies.sh
run make


workdir /opt/kaldi/egs/voxceleb/v1

run echo " \
export train_cmd=run.pl; \
export decode_cmd=run.pl; \
export mkgraph_cmd='run.pl'" > cmd.sh



