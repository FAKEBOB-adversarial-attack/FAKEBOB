#!/bin/bash

set -m

if [[ $(basename $(realpath .)) == "docker" ]]
then
    cd ..
fi

docker build docker -t fakebob
pip3 install gdown
gdown https://drive.google.com/u/0/uc?id=1nEcobGN7_8yyYwdqs1c6XD1UTXqEyhQC&export=download
wait
tar xzf pre-models.tgz

gdown https://drive.google.com/u/0/uc?id=1T_hx9Pqopk-rlmiSrBWdXjl825wjBQVF&export=download
wait
tar xzf data.tar.gz

docker/run.sh python3 build_spk_models.py

echo Setup complete!
