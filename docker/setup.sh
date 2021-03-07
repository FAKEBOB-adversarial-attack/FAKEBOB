#!/bin/bash

#URL_FORM="https://docs.google.com/forms/d/e/1FAIpQLSdQhpq2Be2CktaPhuadUMU7ZDJoQuRlFlzNO45xO-drWQ0AXA/viewform?fbzx=7440236747203254000"
#VOX1_DEV_BASE_LINK="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_parta"
#VOX1_TEST_LINK="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip"
#VOX2_DEV_BASE_LINK="https://www.robots.ox.ac.uk/\~vgg/data/voxceleb/vox1a/vox2_dev_mp4_parta"
#VOX2_TEST_LINK="https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox2_test_aac.zip"
#VOX1_DATA_DIR="./data/vox1"
#VOX2_DATA_DIR="./data/vox2"

#if ! command -v unzip &> /dev/null
#then
#    echo 'Please install the program "unzip" before running the script.'
#    exit
#fi


#printf "\n\n====================\n\n"
#printf "Please request access to the VoxCeleb dataset here: \n$URL_FORM \nPress Enter after you registered.\n"
#read
#echo "Enter your VoxCeleb login information."
#read -p "Username: " username
#read -p "Password: " password
#
#mkdir -p $VOX1_DATA_DIR
#cd $VOX1_DATA_DIR
#
#download_vox_file() {
#    link="$1"
#    wget --user "$username" --password "$password" "$link"
#}
#mkdir dev 
#cd dev
#for e in a b c d
#do
#    download_vox_file "$VOX1_DEV_BASE_LINK""$e"
#done
#cat vox1_dev* > vox1_dev_wav.zip
#unzip *.zip
#cd ..
#
#mkdir test
#cd test
#download_vox_file $VOX1_TEST_LINK
#unzip *.zip
#cd ..
#
#
#cd ../..
#cd $VOX2_DATA_DIR
#mkdir dev
#cd dev
#for e in a b c d e f g h i
#do
#    download_vox_file "$VOX2_DEV_BASE_LINK""$e"
#done                       
#cat vox2_dev_aac* > vox2_aac.zip
#unzip *.zip
#cd ..
#mkdir test
#cd test
#download_vox_file $VOX2_TEST_LINK
#unzip *.zip
#cd ..


#docker run -it -v $PWD:/mounted fakebob bash -ic "
#    cd ./egs/voxceleb/v1;
#    ./run.sh                            
#"


if [[ $(basename $(realpath .)) == "docker" ]]
then
    cd ..
fi

docker build docker -t fakebob
pip3 install gdown
gdown https://drive.google.com/u/0/uc?id=1nEcobGN7_8yyYwdqs1c6XD1UTXqEyhQC&export=download
tar xzf pre-models.tgz

gdown https://drive.google.com/u/0/uc?id=1T_hx9Pqopk-rlmiSrBWdXjl825wjBQVF&export=download
tar xzf data.tar.gz

docker/run.sh python3 build_spk_models.py

echo Setup complete!
