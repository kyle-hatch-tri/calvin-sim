

export CUDA_VISIBLE_DEVICES=0
python3 -u scratch.py &

export CUDA_VISIBLE_DEVICES=1
python3 -u scratch.py &

export CUDA_VISIBLE_DEVICES=2
python3 -u scratch.py 