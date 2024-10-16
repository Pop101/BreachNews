export SINGULARITY_CACHEDIR=/gscratch/stf/lleibm/.cache/.apptainer
export HF_TOKEN=$(cat ~/hf_token)

volume=/gscratch/stf/lleibm/apptainer
model="mistralai/Mistral-7B-Instruct-v0.3"

mkdir -p $volume
apptainer run --nv --bind "$volume:/data" \
    docker://ghcr.io/huggingface/text-generation-inference:2.3.1 \
    --model-id "$model" \
    --port 3023 \
    #--quantize eetq 
    #--max-batch-prefill-tokens=8242 --max-total-tokens=8192 --max-input-tokens=8191