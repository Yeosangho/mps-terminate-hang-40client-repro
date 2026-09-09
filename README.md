# `terminate_client` never returns
<!-- REPRO-CMD -->
## Reproduce

```bash
# One trial. Repeat until it hangs -- roughly 1 in 7.
docker run --rm --gpus '"device=0"' --ipc=host \
  -v "$PWD/scripts:/w" -w /w -e OUT=/w/out \
  -e KIND=queuefill -e KERNEL_SEC=1.0 -e BIG=2048 -e MEM_MB=256 -e DUR=300 \
  -e NVIC=2 -e VBATCH=32 \
  nvcr.io/nvidia/pytorch:26.06-py3 bash /w/run_many.sh 40 40 0
```

A clean trial finishes in about 90 seconds. **A reproducing trial never
finishes**: one `terminate_client` in the middle of the sequence returns
nothing, forever. Confirm it from another shell —

```bash
ps -eo pid,etime,cmd | grep '[n]vidia-cuda-mps-control'   # one with a growing etime
cat /proc/<pid>/task/*/wchan                              # __skb_wait_for_more_packets
```

Kill the trial with `docker kill`; the daemon does not recover on its own.


`nvidia-cuda-mps-control terminate_client <server> <pid>` has no upper bound. Issue
forty of them in sequence against a busy MPS server and one of them — not the first,
not predictably the last — simply does not come back. Not in seconds, not in hours.

```
17:35:55   terminate_client c1  (pid 152)   19 ms  -> 0
...                                          all 17-34 ms
17:35:55   terminate_client c14 (pid 165)   34 ms  -> 0
17:35:55   terminate_client c15 (pid 166)   ---- no response, 30+ minutes ----
```

The control process is alive and sleeping, waiting on the daemon's reply socket:

```
hung nvidia-cuda-mps-control
  main   thread   futex_wait_queue
  worker thread   __skb_wait_for_more_packets     <- blocked in recv, forever
  State: S (sleeping)   Threads: 2
```

Nothing else is wrong with the machine. `dmesg` logs **no Xid**. The GPU sits at
**0 % utilisation** while still holding 45 GB and 30 attached clients. The client
that was asked to terminate is not terminated, and stops logging at the same instant.
The untouched ResNet-152 neighbours stop too — `alive=1`, `err=-`, no exception,
no further output.

## The control daemon is the one that is stuck

Looking at all three parties an hour into the hang:

```
nvidia-cuda-mps-control -d   (the daemon)
  main thread   __skb_wait_for_more_packets    <- the DAEMON is blocked in recv
  thread 2      pipe_read

nvidia-cuda-mps-control      (the hung CLI)
  main thread   futex_wait_queue
  thread 2      __skb_wait_for_more_packets    <- waiting for the daemon's reply

nvidia-cuda-mps-server
  pipe_read                                    <- idle, nothing to do
```

The CLI is not slow. **The daemon never produces a reply because it is itself
blocked receiving**, and the server it fronts is sitting idle in `pipe_read` — so
this is not a GPU operation that has not finished. Something the daemon is waiting
for never arrives, and there is no timeout anywhere in the chain to break it.

Two independent sessions reproduced this on the same node with the same signature:
one hung at c40 of 40 for five hours, the other at c15 of 40 for thirty minutes and
counting. Roughly **1 in 7 cells**.

## Why this matters more than "termination is slow"

Production callers put a timeout around this call and escalate to `SIGKILL` when it
expires. That escalation is the damaging step: `SIGKILL` on a client holding a
resident queue leaves unreclaimable entries in the MPS server, and once those
accumulate every new `cuInit` is refused with

```
Error 807: MPS server is not ready to accept new MPS client requests
```

which is only recoverable by restarting the server. So the visible symptom in
production — "termination took tens of seconds" — is not the defect. The defect is
that it takes *forever*, and the tens of seconds is just where somebody's timeout
cut it off.

Neither of the timeouts that exist applies to this path:

| knob | scope | applies here |
|---|---|---|
| `MPS_TERMINATE_SINGLE_TIMEOUT_SEC=4` | client-side receive | no |
| `MPS_TERM_CLI_TIMEOUT=7` | orchestration agent | no |
| — | `nvidia-cuda-mps-control` itself | **none exists** |

## The count is the ingredient, not the command

The same CLI sequence against a *small* population does not reproduce it. Measured
in Kubernetes with the identical procedure — `ps` then `terminate_client`, issued
sequentially, one call at a time:

| reclaims per cell | cells | hangs |
|---|---|---|
| 2 | 24 | **0** |
| 40 | 14 | **2** |

Two arms were run at the small size, one issuing `ps` before each terminate and one
issuing only the terminate; both were clean, which also rules `ps` out as an
ingredient. What changes the outcome is how many terminations are issued in a row.

## Issue them one at a time

The MPS control daemon accepts **one connection at a time**. Concurrent callers are
not queued — they are refused with `Cannot send command to MPS control daemon
process` in about a millisecond, having done nothing. A harness that fires
terminations in parallel will produce a fast, clean-looking series in which most of
the terminations never happened. The runner counts refusals and reports them
(`refused by the control daemon: 0/40` on a good rep); a nonzero count voids the cell.

## Reading a response

The CLI prints the result as a bare CUresult. **Only `0` is success.**

| response | meaning |
|---|---|
| `0` | terminated |
| `201` | `CUDA_ERROR_INVALID_CONTEXT` |
| `Invalid process <pid>!` | the server does not know that pid |
| `Server 0 not found` | the server pid was never resolved — nothing was issued |
| `Cannot send command to MPS control daemon process` | refused, nothing was issued |

Counting anything but `0` as a reclaim inflates the success rate and, worse, mixes
the ~1 ms cost of a refusal into the latency distribution. In one earlier campaign
only 73 of 99 responses were genuine reclaims.

## Environment

NVIDIA B200 (CC 10.0, 148 SMs, 183 GB), driver 580.x, CUDA MPS with a
private control daemon per rep, `nvcr.io/nvidia/pytorch:26.06-py3`. Evidence from
the second reproduction is in `evidence/` — the hung process's kernel stacks, the
full termination log, and the GPU state while it was hung.
