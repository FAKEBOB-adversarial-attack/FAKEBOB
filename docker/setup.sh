#!/bin/bash

set -e

if [[ $(basename $(realpath .)) == "docker" ]]
then
    cd ..
fi

if ! command -v docker &> /dev/null
then
    echo "Docker doesn't seem to be installed on your system. The docker/setup.sh script uses docker to set up the environment, but can't install docker itself. Please install docker."
    exit
fi

docker build docker -t fakebob
pip3 install gdown
gdown https://drive.google.com/u/0/uc?id=1T_hx9Pqopk-rlmiSrBWdXjl825wjBQVF&export=download
wait
tar xzf pre-models.tgz



gdown https://drive.google.com/u/0/uc?id=1nEcobGN7_8yyYwdqs1c6XD1UTXqEyhQC&export=download
wait
tar xzf data.tar.gz

docker/run.sh python3 build_spk_models.py

echo Setup complete!
