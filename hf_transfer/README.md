# How to run (inside both containers)
Use the IP spark2 uses to reach spark1 (often the same interface as NCCL_SOCKET_IFNAME / RoCE). Start rank 0 first, then rank 1.

```
cd /path/inside/container/to/spark-sglang-stack-dashboard/model_transfer
export MODEL_TRANSFER_TORCH_BACKEND=nccl
# spark1 — model must exist under /data/hf/MODEL
./hf_transfer.sh 0 /data/hf/MODEL /data/hf/MODEL 192.168.100.11
# spark2 — /data/hf/MODEL can be empty; first arg can be a dummy dir
./hf_transfer.sh 1 /tmp/.unused /data/hf/MODEL 192.168.100.11
```

If on spark1 --src and --dest are the same and you still want a full push to spark2:
```
./hf_transfer.sh 0 /data/hf/MODEL /data/hf/MODEL 192.168.x.x --all-files
```

Optional: extra flags are passed through to model_transfer.py, e.g. --master-port 29501.

### Sanity checks
- Same MASTER_PORT on both sides; firewall open between nodes.
- NCCL_DEBUG=INFO (you already have it): confirm IB/RoCE in the log during the run.
- python3 in the container must have PyTorch with CUDA if you use nccl.
