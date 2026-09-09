#!/usr/bin/env bash
# queuefill offenders + ResNet152 victims, terminate_client -- does an untouched
# neighbour take an illegal memory access?
#
# The earlier docker negatives were run before the victim launch existed, so their
# neighbours were all queuefill: the tolerant side, which the harness's own comments say
# reports "no damage" almost regardless. This runs the sensitive side.
#
# The harness needs torch, so it runs INSIDE the pytorch image (the host python has none).
# KIND=queuefill is set for the container so the c* clients are offenders; the victim launch
# overrides it per-process with KIND=resnet.
set -u
# UUID of the target GPU. Get it with:
#   nvidia-smi --query-gpu=index,uuid --format=csv
G0=${GPU_UUID:?set GPU_UUID (e.g. GPU-xxxxxxxx-....)}
# Directory holding this script; mounted into the container as /w.
Q=${Q:-$(cd "$(dirname "$0")" && pwd)}
run(){ # <nclient> <nkill> <tag>
  docker run --rm --runtime=nvidia --ipc=host -e NVIDIA_VISIBLE_DEVICES=$G0 \
    -v $Q:/w -w /w -e OUT=/w/out_$3 \
    -e KIND=queuefill -e KERNEL_SEC=1.0 -e BIG=2048 -e MEM_MB=256 -e DUR=300 \
    -e NVIC=${NVIC:-2} -e VBATCH=${VBATCH:-32} \
    nvcr.io/nvidia/pytorch:26.06-py3 bash /w/run_many.sh $1 $2 0 2>&1 \
    | grep -aE "clients ready|refused|untouched|resnet victims|RESULT|VOID"
}
NC=${NC:-40}; NK=${NK:-4}; REPS=${REPS:-6}
for r in $(seq 1 $REPS); do
  echo "##### qf n$NC k$NK + resnet victims  rep$r"
  run $NC $NK "rn${NC}k${NK}r$r"
done
echo "##### DONE"
