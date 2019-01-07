# Kaggle-Inclusive-Images-Challenge
Solution for [Kaggle Inclusive images Challenge](https://www.kaggle.com/c/inclusive-images-challenge)

## Hardware
- HPC prince cluster
- CPU: Intel Core i7
- GPU: Nvidia Tesla P40 
- RAM: 80GB

## Data
- Download the train [open-images-dataset] (https://www.kaggle.com/c/inclusive-images-challenge#Data-Download-&-Getting-Started) to /train

aws s3 --no-sign-request sync s3://open-images-dataset/train train/

- Download the [inclusive-images-challenge-data] (https://www.kaggle.com/c/inclusive-images-challenge/data) to /

  - kaggle competitions download -c inclusive-images-challenge
  - unzip train_human_labels.csv.zip
  - unzip stage_1_sample_submission.csv.zip
  - unzip stage_2_sample_submission.csv.zip
  - unzip stage_1_test_images.zip -d stage_1_test_images
  - unzip stage_2_images.zip -d stage_2_test_images

## Environment
The code assumes you have atleast one Nvidia GPU with CUDA 9 compatible driver and sufficient memory to store the open-images dataset

## Space Requirement
HDD: ~600 GB for Open Images Dataset

## Train on HPC cluster
sbatch run_ResNet.sh

## Training without HPC
python ResNet_train.py

## Data geneartion for training





