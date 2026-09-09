#!/usr/bin/env python3
"""The MPS tenants whose population is the variable under test.

KIND selects the workload, and that choice matters more than it looks:

  queuefill  (default) keeps QUEUE_DEPTH long kernels OUTSTANDING at all times,
             so the client always holds a resident queue when it is reclaimed.
             This is the OFFENDER: it is what gets terminated.
  resnet     a real ResNet152 training loop. This is the VICTIM, and the choice
             is not cosmetic -- see below.
  plain      a compute loop on a modest allocation, kernels that retire.

The NEIGHBOUR's workload is a decision variable, not scenery. Measured in this
same investigation, with the identical reclaim applied to identical offenders:

    60 consecutive terminate_client, neighbours = ResNet152   -> 2/2 POISONED
    the same, neighbours = a synthetic counter loop           -> 0 damage
    matched control (no terminates at all), ResNet neighbours -> 0/2

So a run whose neighbours are also queuefill is measuring the *tolerant* side and
will report "no damage" almost regardless of what the reclaim does. Production
runs queuefill offenders next to ResNet victims; so does this harness.

Use `queuefill` for anything you intend to conclude from. A `plain` tenant is
*known* not to propagate under any reclaim, so running 40 of them and finding no
damage measures nothing -- 4 would give the same answer. The established
boundary is:

    queuefill x terminate_client             -> harmless
    queuefill x SIGKILL or os._exit(0)       -> siblings get cudaErrorIllegalAddress
    counted-barrier wedge x terminate_client -> co-tenants get it with only 4 clients

A tenant holding a resident queue therefore sits *at* the boundary: safe under
the graceful reclaim, lethal under an abrupt one. That is the workload in which
"does the client count change anything?" is a real question.

Progress is counted DEVICE-SIDE and read without synchronising. A host-side
counter cannot distinguish "the kernel executed" from "the launch was queued and
never dispatched", and a tenant that has been poisoned keeps printing happily
until something forces an error check -- which is how a damaged run gets scored
as healthy.

env:
  KIND         queuefill (default) | plain
  QUEUE_DEPTH  outstanding kernels to keep in flight, queuefill only (default 8)
  KERNEL_SEC   approximate seconds per queued kernel (default 5.0)
  TAG          label in the log
  DUR          seconds to run (default 900)
  MEM_MB       device allocation to hold (default 256)
  WORK         matmul edge size (default 512); keep small, this is not a stressor
  REPORT_EVERY seconds between progress lines (default 1.0)
"""
import os
import sys
import time

import torch

KIND = os.environ.get("KIND", "queuefill")
TAG = os.environ.get("TAG", "t")
DUR = float(os.environ.get("DUR", "900"))
MEM_MB = int(os.environ.get("MEM_MB", "256"))
WORK = int(os.environ.get("WORK", "512"))
REPORT_EVERY = float(os.environ.get("REPORT_EVERY", "1.0"))


def say(*a):
    print("[%s]" % TAG, *a, flush=True)


def run_resnet(dev, dev_ctr, one, pinned):
    """The sensitive neighbour: a real cuDNN/cuBLAS training loop."""
    import torchvision
    m = torchvision.models.resnet152(weights=None).to(dev).train()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    lf = torch.nn.CrossEntropyLoss()
    x = torch.randn(int(os.environ.get("BATCH", "32")), 3, 224, 224, device=dev)
    y = torch.randint(0, 1000, (x.shape[0],), device=dev)
    say("ready pid=%d kind=resnet batch=%d" % (os.getpid(), x.shape[0]))
    t0 = time.time(); last = 0.0
    while time.time() - t0 < DUR:
        try:
            opt.zero_grad(set_to_none=True)
            lf(m(x), y).backward()
            opt.step()
            dev_ctr.add_(one)
            pinned.copy_(dev_ctr, non_blocking=True)
        except Exception as exc:  # noqa: BLE001
            say("POISONED exception: %s" % str(exc).replace("\n", " ")[:300])
            return 43
        now = time.time()
        if now - last >= REPORT_EVERY:
            last = now
            say("%.3f probe start=%d alive=1 err=-" % (now, int(pinned[0].item())))
    say("done")
    return 0


def main():
    torch.cuda.init()
    dev = torch.device("cuda:0")

    # a modest resident allocation, so the client owns some VA to tear down
    ballast = torch.empty(int(MEM_MB * 1024 * 1024 // 4), dtype=torch.float32, device=dev)
    ballast.zero_()

    a = torch.randn(WORK, WORK, device=dev)
    b = torch.randn(WORK, WORK, device=dev)

    # queuefill: size one matmul so it takes roughly KERNEL_SEC, then keep
    # QUEUE_DEPTH of them outstanding. The client is then always holding a
    # resident queue at the moment it is reclaimed -- which is the state in
    # which an abrupt ending poisons the siblings.
    QDEPTH = int(os.environ.get("QUEUE_DEPTH", "8"))
    KSEC = float(os.environ.get("KERNEL_SEC", "5.0"))
    if KIND == "queuefill":
        big = int(os.environ.get("BIG", "4096"))
        qa = torch.randn(big, big, device=dev)
        qb = torch.randn(big, big, device=dev)
        # calibrate: how many matmuls fit in KERNEL_SEC
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            qa = torch.mm(qa, qb) * 0.0 + qa
        torch.cuda.synchronize()
        per = max((time.time() - t0) / 3.0, 1e-4)
        REP = max(1, int(KSEC / per))
        say("queuefill calibrated: %.4fs/mm -> %d mm per queued unit, depth=%d"
            % (per, REP, QDEPTH))

    # device-side progress counter + pinned staging, read WITHOUT synchronising
    dev_ctr = torch.zeros(1, dtype=torch.int64, device=dev)
    one = torch.ones(1, dtype=torch.int64, device=dev)
    pinned = torch.zeros(1, dtype=torch.int64).pin_memory()

    if KIND == "resnet":
        return run_resnet(dev, dev_ctr, one, pinned)

    say("ready pid=%d kind=%s mem_mb=%d" % (os.getpid(), KIND, MEM_MB))

    inflight = []          # queuefill: outstanding work markers
    t0 = time.time()
    last = 0.0
    err = "-"
    while time.time() - t0 < DUR:
        try:
            if KIND == "queuefill":
                # Top the queue back up to QUEUE_DEPTH without ever draining it.
                # `query()` tells us how much has retired without blocking; a
                # synchronize here would empty the queue and destroy the very
                # state under test.
                while len(inflight) < QDEPTH:
                    ev = torch.cuda.Event()
                    for _ in range(REP):
                        qa = torch.mm(qa, qb) * 0.0 + qa
                    dev_ctr.add_(one)
                    ev.record()
                    inflight.append(ev)
                while inflight and inflight[0].query():
                    inflight.pop(0)
            else:
                for _ in range(20):
                    a = torch.mm(a, b) * 0.0 + a      # keep values bounded
                    dev_ctr.add_(one)
            pinned.copy_(dev_ctr, non_blocking=True)

            # A poisoned context does not necessarily raise on the next launch,
            # so ask explicitly -- but the probe itself must not be the thing
            # that kills the tenant. `cudaPeekAtLastError` is not exposed on
            # every torch build (measured: it is missing on 26.06, and using it
            # unguarded killed all 36 bystanders and was very nearly read as
            # propagation). `Stream.query()` is non-blocking, present
            # everywhere, and raises on a sticky error.
            if not torch.cuda.current_stream().query():
                pass  # simply still busy; not an error
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).replace("\n", " ")[:300]
            say("POISONED exception: %s" % msg)
            return 43

        now = time.time()
        if now - last >= REPORT_EVERY:
            last = now
            say("%.3f probe start=%d alive=1 err=%s"
                % (now, int(pinned[0].item()), err))
    say("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
