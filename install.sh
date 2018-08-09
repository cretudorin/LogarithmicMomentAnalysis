#!/bin/bash

sudo apt install python-pip -y
sudo apt install virtualenv -y
sudo apt install python-matplotlib -y
sudo apt install python-tk -y

rm venv -r

/bin/bash  -c "virtualenv venv --no-site-packages"
/bin/bash  -c ". ./venv/bin/activate"
pip install -r requirements.txt