if [ -f /etc/bash_completion ]; then source /etc/bash_completion; fi
export TRANSFORMERS_CACHE=/workspace/.hf-cache
export HF_HOME=/workspace/.hf-cache
export TORCHINDUCTOR_CACHE_DIR=/workspace/.torch-cache
echo
echo "Welcome to this Docker container, type \"make help\" to get some help"
echo
