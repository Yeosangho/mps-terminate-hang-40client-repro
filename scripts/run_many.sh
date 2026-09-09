#!/usr/bin/env bash
# run_many.sh -- does terminate_client damage bystanders once the client
# population is large?
#
#   ./run_many.sh <NCLIENT> <NKILL> <GPU>
#     NCLIENT  how many ordinary MPS clients to run  (default 40)
#     NKILL    how many of them to terminate_client  (default 4)
#     GPU      GPU index                             (default 0)
#
# Everything runs against a PRIVATE MPS daemon in a scratch pipe directory.
#
# TWO RULES THIS HARNESS EXISTS TO ENFORCE
#
# 1. Terminates are issued SEQUENTIALLY, and every response is checked.
#    The MPS control daemon accepts ONE connection at a time; concurrent callers
#    get "Cannot send command to MPS control daemon process" in a millisecond
#    without anything happening. A concurrent version of this script produced a
#    fast, clean-looking series in which 31 of 32 terminates had been refused.
#    Refusals are counted and VOID the cell.
#
# 2. Only clients that were NOT terminated are scored.
#    A terminated client's own cudaErrorMpsClientTerminated is its intended
#    ending, not propagation. Counting it turns a clean run into an apparent
#    blast radius.
set -u
NCLIENT=${1:-40}; NKILL=${2:-4}; GPU=${3:-0}
# NVIC ResNet152 victims run alongside and are NEVER terminated. The neighbour's
# workload is a decision variable: the same reclaim that leaves a synthetic
# counter loop untouched kills a cuDNN training loop. A run whose neighbours are
# also queuefill measures the tolerant side and reports "no damage" almost
# regardless of what the reclaim does.
NVIC=${NVIC:-2}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HERE/out.n$NCLIENT.k$NKILL.$$}; mkdir -p "$OUT"
DUR=${DUR:-900}
MEM_MB=${MEM_MB:-256}
SETTLE=${SETTLE:-25}
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_MPS_PIPE_DIRECTORY=$OUT/mps
export CUDA_MPS_LOG_DIRECTORY=$OUT/mpslog
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

declare -a PID
cleanup(){
  for p in ${PID[@]:-} ${VPID[@]:-}; do kill -9 "$p" 2>/dev/null; done
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
}
trap cleanup EXIT

say "private MPS daemon; NCLIENT=$NCLIENT NKILL=$NKILL MEM_MB=$MEM_MB"
nvidia-cuda-mps-control -d || { say "FATAL: cannot start MPS"; exit 1; }
sleep 3

for i in $(seq 1 "$NCLIENT"); do
  env TAG="c$i" DUR="$DUR" MEM_MB="$MEM_MB" \
    python3 "$HERE/tenant.py" > "$OUT/c$i.log" 2>&1 &
  PID[$i]=$!
done

# the sensitive neighbours the rest of this script already gates on, cleans up and
# scores - they were never actually launched
declare -a VPID
for i in $(seq 1 "$NVIC"); do
  env TAG="v$i" DUR="$DUR" KIND=resnet BATCH="${VBATCH:-32}" \
    python3 "$HERE/tenant.py" > "$OUT/v$i.log" 2>&1 &
  VPID[$i]=$!
done

# READINESS GATE. A client that never attached is not a bystander that survived.
NREADY=0
for t in $(seq 1 180); do
  NREADY=0
  for i in $(seq 1 "$NCLIENT"); do
    grep -qa "\] ready " "$OUT/c$i.log" && NREADY=$((NREADY+1))
  done
  [ "$NREADY" -ge "$NCLIENT" ] && break
  sleep 2
done
NVREADY=0
for t in $(seq 1 240); do
  NVREADY=0
  for i in $(seq 1 "$NVIC"); do grep -qa "\] ready " "$OUT/v$i.log" && NVREADY=$((NVREADY+1)); done
  [ "$NVREADY" -ge "$NVIC" ] && break
  sleep 2
done
say "clients ready: $NREADY/$NCLIENT   resnet victims ready: $NVREADY/$NVIC"
if [ "$NVREADY" -lt "$NVIC" ]; then
  say "VOID: only $NVREADY/$NVIC resnet victims attached"
  exit 2
fi
if [ "$NREADY" -lt "$NCLIENT" ]; then
  say "VOID: only $NREADY/$NCLIENT clients attached -- MPS has a client limit"
  say "      (48 on Volta+ by default); lower NCLIENT or raise the limit."
  exit 2
fi

sleep "$SETTLE"

SRV=$(echo get_server_list | nvidia-cuda-mps-control | tr -dc '0-9 \n' | awk '{print $1}' | head -1)
say "mps server pid=$SRV"
[ -z "$SRV" ] && { say "VOID: no MPS server"; exit 2; }

# progress of every client just before the reclaim
cstart(){ grep -a " probe " "$OUT/c$1.log" | tail -1 | grep -oE 'start=[0-9]+' | cut -d= -f2; }
declare -A P0
for i in $(seq 1 "$NCLIENT"); do P0[$i]=$(cstart "$i"); done

FIRED=""
NREJ=0
for k in $(seq 1 "$NKILL"); do
  c0=$(date +%s%3N)
  resp=$(echo "terminate_client $SRV ${PID[$k]}" | nvidia-cuda-mps-control 2>&1)
  c1=$(date +%s%3N)
  say "  terminate_client c$k (pid ${PID[$k]}) $((c1-c0)) ms -> ${resp:-<empty>}"
  echo "$resp" | grep -qa "Cannot send command" && NREJ=$((NREJ+1))
  FIRED="$FIRED $k"
done
say "  refused by the control daemon: $NREJ/$NKILL"
if [ "$NREJ" -gt 0 ]; then
  say "VOID: $NREJ terminates were REFUSED, not served -- nothing was reclaimed"
  exit 2
fi

sleep 30

# ---- score ONLY the clients nobody touched ---------------------------------
NILL=0; NERR=0; NDEAD=0; NSTALL=0; NSCORED=0; DET=""
for i in $(seq 1 "$NCLIENT"); do
  case " $FIRED " in *" $i "*) continue ;; esac
  NSCORED=$((NSCORED+1))
  ln=$(grep -aoE "cudaErrorIllegalAddress|illegal memory access|cudaErrorMpsRpcFailure|cudaErrorMpsClientTerminated|cudaError=[0-9]+|POISONED[^$]*" "$OUT/c$i.log" | tail -1)
  if [ -n "$ln" ]; then
    NERR=$((NERR+1)); DET="$DET c$i=[$ln]"
    echo "$ln" | grep -qaE "IllegalAddress|illegal memory" && NILL=$((NILL+1))
  fi
  if ! kill -0 "${PID[$i]}" 2>/dev/null; then NDEAD=$((NDEAD+1)); continue; fi
  p1=$(cstart "$i")
  if [ -n "${P0[$i]}" ] && [ -n "$p1" ] && [ "$p1" -le "${P0[$i]}" ]; then
    NSTALL=$((NSTALL+1)); DET="$DET c$i=STALLED"
  fi
done

say "  --- untouched clients: $NSCORED scored ---"
[ -n "$DET" ] && say "   $DET"
# ---- the ResNet victims: the sensitive neighbours -------------------------
VDEAD=0; VILL=0; VERR=0; VDET=""
for i in $(seq 1 "$NVIC"); do
  ln=$(grep -aoE "cudaErrorIllegalAddress|illegal memory access|CUDNN_STATUS_[A-Z_]+|CUBLAS_STATUS_[A-Z_]+|cudaErrorMpsRpcFailure|POISONED[^$]{0,80}" "$OUT/v$i.log" | tail -1)
  if [ -n "$ln" ]; then
    VERR=$((VERR+1)); VDET="$VDET v$i=[$ln]"
    echo "$ln" | grep -qaE "IllegalAddress|illegal memory" && VILL=$((VILL+1))
  fi
  kill -0 "${VPID[$i]}" 2>/dev/null || VDEAD=$((VDEAD+1))
done
say "  --- resnet victims (never terminated): $NVIC ---"
[ -n "$VDET" ] && say "   $VDET"

say "RESULT nclient=$NCLIENT nkill=$NKILL untouched=$NSCORED dead=$NDEAD stalled=$NSTALL with_error=$NERR illegal=$NILL  resnet=$NVIC vdead=$VDEAD verr=$VERR villegal=$VILL"
say "logs in $OUT"
