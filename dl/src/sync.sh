#/usr/bin/env bash

IP1=$1
export IP=$IP1
rsync -avz . \
    -e "ssh -i /Users/krishnakalyan3/MOOC/MachineLearning/key/crime.pem" \
    ubuntu@$IP:/home/ubuntu/
