
python -m pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl 

python -m pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl 

Cd apex 
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

export Torch_DIR=/home/wzha8158/PCDet/lib/python3.6/site-packages/torch

rm build
sudo apt-get install libboost-all-dev
Cd spconv 
python3.6 setup.py bdist_wheel

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ xenial main'
sudo apt update
sudo apt install cmake