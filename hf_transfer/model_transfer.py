#!/usr/bin/env python3
"""
High-speed Hugging Face model transfer using RDMA/RoCE or parallel TCP.

Supports:
- RDMA via PyTorch's distributed backend (NCCL/GLOO)
- Parallel TCP streaming for fallback
- Directory synchronization with checksums
- Resume capability for interrupted transfers
"""

import argparse
import hashlib
import json
import multiprocessing
import os
import shlex
import shutil
import socket
import subprocess
import struct
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle

import numpy as np

# PyTorch is required for --mode rdma (distributed transfer).
_TORCH_IMPORT_ERROR: Optional[ImportError] = None
try:
    import torch
    import torch.distributed as dist
    RDMA_AVAILABLE = True
except ImportError as e:
    RDMA_AVAILABLE = False
    _TORCH_IMPORT_ERROR = e

from concurrent.futures import ThreadPoolExecutor, as_completed


def _import_zmq():
    """Load pyzmq only when parallel TCP is used (optional dependency)."""
    try:
        import zmq
        return zmq
    except ImportError as e:
        raise ImportError(
            "parallel TCP mode requires pyzmq. Install with: pip install pyzmq"
        ) from e

DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB chunks
DEFAULT_NUM_STREAMS = 16  # Parallel streams for TCP fallback

_ENV_PREFIX = "MODEL_TRANSFER_"
_DISK_HEARTBEAT_SEC = 30.0


def _disk_heartbeat_transfer(stop: threading.Event, watch_dir: str) -> None:
    """Periodic free-space lines during long RDMA transfers (multi-GB single-file sends)."""
    while not stop.wait(_DISK_HEARTBEAT_SEC):
        try:
            usage = shutil.disk_usage(watch_dir)
            free_gib = usage.free / (1024**3)
            total_gib = usage.total / (1024**3)
            print(
                f"[heartbeat] disk free {free_gib:.2f} GiB of {total_gib:.2f} GiB ({watch_dir})",
                flush=True,
            )
        except OSError as e:
            print(f"[heartbeat] disk_usage failed: {e}", flush=True, file=sys.stderr)


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    script_dir = Path(__file__).resolve().parent
    load_dotenv(script_dir / ".env")


def _mt_env(key: str) -> Optional[str]:
    v = os.environ.get(f"{_ENV_PREFIX}{key}")
    return v if v else None


def _mt_env_int(key: str, default: int) -> int:
    v = _mt_env(key)
    if v is None:
        return default
    return int(v)


def _resolve_dist_backend() -> str:
    """gloo = CPU tensors (works without NCCL). nccl = GPU tensors (needs CUDA + NCCL)."""
    raw = (_mt_env('TORCH_BACKEND') or 'gloo').strip().lower()
    if raw not in ('gloo', 'nccl'):
        print(f'Warning: unknown MODEL_TRANSFER_TORCH_BACKEND={raw!r}, using gloo')
        raw = 'gloo'
    if raw == 'nccl' and not torch.cuda.is_available():
        print('Warning: MODEL_TRANSFER_TORCH_BACKEND=nccl but CUDA is not available; using gloo.')
        raw = 'gloo'
    return raw


@dataclass
class FileInfo:
    """Metadata for a file to transfer."""
    path: str
    size: int
    mtime: float
    md5: Optional[str] = None
    
    def compute_md5(self, root: Path) -> Optional[str]:
        """Compute MD5 hash of file."""
        full_path = root / self.path
        if not full_path.is_file():
            return None
        
        md5 = hashlib.md5()
        with open(full_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()


class RDMATransfer:
    """PyTorch distributed transfer (Gloo or NCCL)."""
    
    def __init__(self, rank: int, world_size: int, master_addr: str, master_port: int):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = _resolve_dist_backend()
        self.device = torch.device('cuda' if self.backend == 'nccl' else 'cpu')
        
        # Initialize distributed environment
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(master_port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        
        timeout_sec = _mt_env_int('INIT_TIMEOUT_SEC', 1800)
        print(
            f'PyTorch distributed: backend={self.backend}, device={self.device} '
            f'(set MODEL_TRANSFER_TORCH_BACKEND=gloo|nccl; gloo needs no NCCL)',
            flush=True,
        )
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_sec),
        )
        
    def send_file(self, file_path: str, dest_path: str):
        """Send file using collective broadcast."""
        dev = self.device
        if self.rank == 0:  # Sender
            with open(file_path, 'rb') as f:
                data = f.read()
            n = len(data)
            # Avoid list(data): one Python int per byte makes multi-GB sends impractically slow.
            # torch.tensor copies from the buffer (writable tensor); torch.as_tensor(np.frombuffer)
            # would alias read-only memory and trigger PyTorch warnings / undefined behavior.
            tensor_cpu = torch.tensor(
                np.frombuffer(data, dtype=np.uint8), dtype=torch.uint8
            )
            del data
            tensor = tensor_cpu.to(dev, non_blocking=(dev.type == 'cuda'))
            size_tensor = torch.tensor([n], dtype=torch.long, device=dev)
            dist.broadcast(size_tensor, src=0)
            dist.broadcast(tensor, src=0)
            print(f"Sent {file_path} ({n / 1e9:.2f} GB)", flush=True)
        else:  # Receiver
            size_tensor = torch.zeros(1, dtype=torch.long, device=dev)
            dist.broadcast(size_tensor, src=0)
            size = int(size_tensor.item())
            
            tensor = torch.empty(size, dtype=torch.uint8, device=dev)
            dist.broadcast(tensor, src=0)
            
            os.makedirs(os.path.dirname(dest_path) or '.', exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(tensor.detach().cpu().numpy().tobytes())
            print(f"Received {dest_path} ({size / 1e9:.2f} GB)", flush=True)
    
    def send_directory(self, src_dir: str, dest_dir: str, files: List[FileInfo]):
        """Send multiple files sequentially."""
        for file_info in files:
            src_path = Path(src_dir) / file_info.path
            dest_path = Path(dest_dir) / file_info.path
            self.send_file(str(src_path), str(dest_path))
    
    def cleanup(self):
        """Clean up distributed resources."""
        dist.destroy_process_group()


class ParallelTCPTransfer:
    """High-speed parallel TCP transfer with file chunking."""
    
    def __init__(self, local_ip: str, peer_ip: str, port: int, num_streams: int = DEFAULT_NUM_STREAMS):
        self.local_ip = local_ip
        self.peer_ip = peer_ip
        self.port = port
        self.num_streams = num_streams
        self._zmq = _import_zmq()
        self.context = self._zmq.Context()

    def _send_chunk(self, chunk_id: int, data: bytes, offset: int, file_path: str):
        """Send a single chunk over a specific stream."""
        zmq = self._zmq
        socket = self.context.socket(zmq.PUSH)
        socket.connect(f"tcp://{self.peer_ip}:{self.port + chunk_id % self.num_streams}")
        
        # Send metadata: [file_path_len, file_path, offset, chunk_size, data]
        file_path_encoded = file_path.encode('utf-8')
        metadata = struct.pack('!I', len(file_path_encoded)) + file_path_encoded
        metadata += struct.pack('!QQ', offset, len(data))
        
        socket.send(metadata, zmq.SNDMORE)
        socket.send(data)
        socket.close()
    
    def send_file_parallel(self, file_path: str, dest_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """Send a file using parallel streams over different ports."""
        file_size = os.path.getsize(file_path)
        chunks = []
        
        # Split file into chunks
        with open(file_path, 'rb') as f:
            offset = 0
            while offset < file_size:
                chunk_data = f.read(chunk_size)
                chunks.append((offset, chunk_data))
                offset += len(chunk_data)
        
        # Send chunks in parallel
        with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
            futures = []
            for i, (offset, data) in enumerate(chunks):
                future = executor.submit(self._send_chunk, i, data, offset, dest_path)
                futures.append(future)
            
            # Wait for all chunks
            for future in as_completed(futures):
                future.result()
        
        print(f"Sent {file_path} ({file_size / 1e9:.2f} GB) using {len(chunks)} chunks")
    
    def receive_file(self, output_dir: str):
        """Receive file from multiple streams and reassemble."""
        receivers = []
        
        def receive_stream(stream_id: int):
            zmq = self._zmq
            socket = self.context.socket(zmq.PULL)
            socket.bind(f"tcp://*:{self.port + stream_id}")
            
            while True:
                # Receive metadata
                metadata = socket.recv()
                if not metadata:
                    break
                
                # Parse metadata
                offset = 0
                file_path_len = struct.unpack('!I', metadata[:4])[0]
                file_path = metadata[4:4+file_path_len].decode('utf-8')
                offset = struct.unpack('!Q', metadata[4+file_path_len:4+file_path_len+8])[0]
                chunk_size = struct.unpack('!Q', metadata[4+file_path_len+8:4+file_path_len+16])[0]
                
                # Receive data
                data = socket.recv()
                
                # Write chunk to file
                full_path = Path(output_dir) / file_path
                os.makedirs(full_path.parent, exist_ok=True)
                with open(full_path, 'rb+') if full_path.exists() else open(full_path, 'wb') as f:
                    f.seek(offset)
                    f.write(data)
                
                print(f"Received chunk {offset} for {file_path}")
            
            socket.close()
        
        # Start receiver threads
        threads = []
        for i in range(self.num_streams):
            t = threading.Thread(target=receive_stream, args=(i,))
            t.start()
            threads.append(t)
        
        # Keep running until stopped
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            pass


class ModelTransfer:
    """Main orchestrator for model transfers."""
    
    def __init__(self, src_dir: str, dest_dir: str, use_rdma: bool = True):
        self.src_dir = Path(src_dir).resolve()
        self.dest_dir = Path(dest_dir).resolve()
        self.use_rdma = use_rdma and RDMA_AVAILABLE
        
        # Scan source directory
        self.files = self._scan_directory()
        
    def _scan_directory(self) -> List[FileInfo]:
        """Scan directory for all files."""
        if not self.src_dir.is_dir():
            print(
                f"Note: --src is not a directory ({self.src_dir}); "
                "using empty file list (normal on rank>0 when only the sender has the model)."
            )
            return []
        files = []
        for root, _, filenames in os.walk(self.src_dir):
            for filename in filenames:
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(self.src_dir)
                
                file_info = FileInfo(
                    path=str(rel_path),
                    size=file_path.stat().st_size,
                    mtime=file_path.stat().st_mtime
                )
                files.append(file_info)
        
        # Sort by size (largest first for better parallelism)
        files.sort(key=lambda x: x.size, reverse=True)
        print(f"Found {len(files)} files, total size: {sum(f.size for f in files) / 1e9:.2f} GB")
        return files
    
    def _check_existing(self, dest_root: Path) -> List[FileInfo]:
        """Check which files already exist at destination."""
        to_transfer = []
        for file_info in self.files:
            dest_path = dest_root / file_info.path
            if not dest_path.exists() or dest_path.stat().st_size != file_info.size:
                to_transfer.append(file_info)
            else:
                print(f"Skipping {file_info.path} (already exists)")
        return to_transfer
    
    def transfer_rsync(self, dest_host: str, dest_user: str = None):
        """Fallback to rsync for compatibility."""
        remote_target = f"{dest_user}@{dest_host}" if dest_user else dest_host
        dest = f"{remote_target}:{self.dest_dir}" if dest_user else f"{dest_host}:{self.dest_dir}"
        mkdir_cmd = ["ssh", remote_target, "mkdir", "-p", str(self.dest_dir)]
        print(f"Running: {' '.join(shlex.quote(x) for x in mkdir_cmd)}")
        if subprocess.run(mkdir_cmd).returncode != 0:
            print("Error: could not create destination path on remote host (check SSH access).")
            return
        cmd = f"rsync -avz --progress {self.src_dir}/ {dest}/"
        print(f"Running: {cmd}")
        os.system(cmd)
    
    def transfer_parallel_tcp(self, local_ip: str, peer_ip: str, port: int, num_streams: int = DEFAULT_NUM_STREAMS):
        """Transfer using parallel TCP."""
        # Check existing files first
        to_transfer = self._check_existing(self.dest_dir)
        if not to_transfer:
            print("All files already exist at destination")
            return
        
        print(f"Transferring {len(to_transfer)} files using parallel TCP...")
        
        # Start receiver on peer (this would be run on spark2)
        # For simplicity, we'll do sequential file transfer
        transfer = ParallelTCPTransfer(local_ip, peer_ip, port, num_streams)
        
        # For each file, transfer
        for file_info in to_transfer:
            src_path = self.src_dir / file_info.path
            dest_path = self.dest_dir / file_info.path
            
            print(f"Transferring {file_info.path} ({file_info.size / 1e9:.2f} GB)")
            transfer.send_file_parallel(str(src_path), str(dest_path))
    
    def transfer_rdma(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
        *,
        sender_send_all: bool = False,
    ):
        """Transfer using PyTorch distributed (Gloo or NCCL).

        Rank 0 builds the file list and broadcasts a (path, size) manifest so rank>0
        does not need a local copy of --src (typical for Docker: model only on master).
        """
        if not self.use_rdma:
            print("PyTorch is not available. Install with: pip install torch")
            if _TORCH_IMPORT_ERROR is not None:
                print(f"Import error: {_TORCH_IMPORT_ERROR}")
            return False

        if rank == 0 and not self.files:
            print("Error: no files under --src on rank 0 (check path and permissions).")
            return False

        stop_hb = threading.Event()
        watch_dir = str(self.src_dir) if rank == 0 else str(self.dest_dir)
        hb_thread = threading.Thread(
            target=_disk_heartbeat_transfer,
            args=(stop_hb, watch_dir),
            name="disk-heartbeat",
            daemon=True,
        )
        hb_thread.start()
        try:
            rdma = RDMATransfer(rank, world_size, master_addr, master_port)
            try:
                if rank == 0:
                    if sender_send_all:
                        to_transfer = list(self.files)
                    else:
                        to_transfer = self._check_existing(self.dest_dir)
                    manifest: List[Tuple[str, int]] = [(f.path, f.size) for f in to_transfer]
                    object_list = [manifest]
                else:
                    object_list = [None]

                dist.broadcast_object_list(object_list, src=0)
                manifest = object_list[0]
                assert manifest is not None

                if not manifest:
                    if rank == 0:
                        print("Nothing to transfer (all files already at destination, or empty manifest).")
                    else:
                        print("Receiver: manifest is empty; nothing to do.")
                    return True

                if rank == 0:
                    print(f"Transferring {len(manifest)} files via {rdma.backend}...", flush=True)
                else:
                    print(f"Receiving {len(manifest)} files into {self.dest_dir}...", flush=True)

                for path, _size in manifest:
                    src_path = self.src_dir / path
                    dest_path = self.dest_dir / path
                    rdma.send_file(str(src_path), str(dest_path))

                return True
            finally:
                rdma.cleanup()
        finally:
            stop_hb.set()
            hb_thread.join(timeout=5.0)
    
    def transfer_with_resume(self, method: str = 'rdma', **kwargs):
        """Transfer with resume capability."""
        # Save transfer state
        state_file = self.dest_dir / '.transfer_state.json'
        
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                # Resume from last state
                completed = set(state.get('completed', []))
                self.files = [f for f in self.files if f.path not in completed]
        
        # Perform transfer
        if method == 'rdma':
            success = self.transfer_rdma(**kwargs)
        elif method == 'parallel_tcp':
            self.transfer_parallel_tcp(**kwargs)
            success = True
        else:
            success = False
        
        # Save state on success
        if success:
            completed = [f.path for f in self.files]
            with open(state_file, 'w') as f:
                json.dump({'completed': completed, 'timestamp': time.time()}, f)
        
        return success


def main():
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description='High-speed Hugging Face model transfer',
        epilog=(
            'Optional defaults from .env (see env.example): '
            f'{_ENV_PREFIX}SRC, {_ENV_PREFIX}DEST, {_ENV_PREFIX}MODE, '
            f'{_ENV_PREFIX}RANK, {_ENV_PREFIX}WORLD_SIZE, {_ENV_PREFIX}MASTER_ADDR, '
            f'{_ENV_PREFIX}MASTER_PORT, {_ENV_PREFIX}LOCAL_IP, {_ENV_PREFIX}PEER_IP, '
            f'{_ENV_PREFIX}ZMQ_PORT, {_ENV_PREFIX}NUM_STREAMS, {_ENV_PREFIX}DEST_HOST, '
            f'{_ENV_PREFIX}DEST_USER, {_ENV_PREFIX}TORCH_BACKEND, {_ENV_PREFIX}INIT_TIMEOUT_SEC'
        ),
    )
    parser.add_argument('--src', default=_mt_env('SRC'), help='Source directory (spark1)')
    parser.add_argument('--dest', default=_mt_env('DEST'), help='Destination directory (spark2)')
    parser.add_argument(
        '--mode',
        choices=['rdma', 'parallel_tcp', 'rsync'],
        default=_mt_env('MODE') or 'rdma',
        help='Transfer mode',
    )

    # RDMA options
    parser.add_argument(
        '--rank',
        type=int,
        default=_mt_env_int('RANK', 0),
        help='Rank (0 for sender, 1 for receiver)',
    )
    parser.add_argument(
        '--world-size',
        type=int,
        default=_mt_env_int('WORLD_SIZE', 2),
        help='World size (usually 2)',
    )
    parser.add_argument(
        '--master-addr',
        default=_mt_env('MASTER_ADDR') or '192.168.100.11',
        help='Master address',
    )
    parser.add_argument(
        '--master-port',
        type=int,
        default=_mt_env_int('MASTER_PORT', 29500),
        help='Master port',
    )
    parser.add_argument(
        '--all-files',
        action='store_true',
        help=(
            'Rank 0 only: send every file under --src without skipping files that already '
            'exist at --dest (use when --src and --dest are the same path and you still '
            'want to push to rank>0).'
        ),
    )

    # Parallel TCP options
    parser.add_argument(
        '--local-ip',
        default=_mt_env('LOCAL_IP') or '192.168.100.11',
        help='Local IP for TCP',
    )
    parser.add_argument(
        '--peer-ip',
        default=_mt_env('PEER_IP') or '192.168.100.12',
        help='Peer IP for TCP',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=_mt_env_int('ZMQ_PORT', 5555),
        help='Base port for TCP streams',
    )
    parser.add_argument(
        '--num-streams',
        type=int,
        default=_mt_env_int('NUM_STREAMS', DEFAULT_NUM_STREAMS),
        help='Number of parallel streams',
    )

    # Rsync options
    parser.add_argument('--dest-host', default=_mt_env('DEST_HOST'), help='Destination host for rsync')
    parser.add_argument(
        '--dest-user',
        default=_mt_env('DEST_USER') or 'chenchen',
        help='Destination user for rsync',
    )

    args = parser.parse_args()

    if args.mode == 'rdma' and args.rank == 0:
        if not Path(args.src).is_dir():
            parser.error(f'--src must exist and be a directory on rank 0: {args.src!r}')

    if not args.src or not args.dest:
        parser.error(
            f'Source and destination are required (use --src/--dest or '
            f'{_ENV_PREFIX}SRC / {_ENV_PREFIX}DEST in .env)'
        )

    if args.mode == 'rdma' and not RDMA_AVAILABLE:
        print(
            'Error: --mode rdma requires PyTorch. Install in this environment:\n'
            '  pip install torch\n'
            'If you use CUDA, install the wheel that matches your system from https://pytorch.org/'
        )
        if _TORCH_IMPORT_ERROR is not None:
            print(f'Details: {_TORCH_IMPORT_ERROR}')
        sys.exit(1)

    transfer = ModelTransfer(args.src, args.dest, use_rdma=(args.mode == 'rdma'))
    
    if args.mode == 'rdma':
        success = transfer.transfer_rdma(
            rank=args.rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
            sender_send_all=args.all_files,
        )
    elif args.mode == 'parallel_tcp':
        transfer.transfer_parallel_tcp(
            local_ip=args.local_ip,
            peer_ip=args.peer_ip,
            port=args.port,
            num_streams=args.num_streams
        )
        success = True
    else:  # rsync
        if not args.dest_host:
            print("Error: --dest-host required for rsync mode")
            sys.exit(1)
        transfer.transfer_rsync(args.dest_host, args.dest_user)
        success = True
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
