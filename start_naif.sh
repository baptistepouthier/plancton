python3 -m virtualenv virtual_env_py3_naif

cd virtual_env_py3_naif

source bin/activate

cp ../gen_features_* .

module load cuda/8.0


module load cudnn/5.1-cuda-8.0


#export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp34-cp34m-linux_x86_64.whl

export TF_BINARY_URL=https://pypi.python.org/packages/d1/ac/4cbd5884ec518eb1064ebafed86fcbcff02994b971973d9e4d1c3fe72204/tensorflow_gpu-1.2.0-cp34-cp34m-manylinux1_x86_64.whl#md5=69c299e9af480a1f570279d7e8a0b959

#pip3 install --upgrade pip


pip3 install $TF_BINARY_URL

pip3 install keras h5py pillow



#pip3 install $TF_BINARY_URL



cd /home/bpouthie/Documents/partage/confusion
python3 test.py
