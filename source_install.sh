pip install -r requirements.txt
export MAX_JOBS=10
# for 5090(reduce arch list)
export FLASH_ATTN_CUDA_ARCHS="80;120"
pip install flash-attn==2.8.3 --no-build-isolation
