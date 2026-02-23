import sys
import os
from typing import Dict, List, Tuple
import gc
from tqdm import tqdm
import torch
from queue import Empty
import multiprocessing as mp
from .. import logger
from ..processor import _device
from . import RodnaProcessor


def _detect_cuda_devices() -> Dict[str, int]:
    result = {}
    count = torch.cuda.device_count()
    logger.info(f"Found [{count}] CUDA device(s)")

    for i in range(count):
        d_name = torch.cuda.get_device_name(i)
        d_props = torch.cuda.get_device_properties(i)
        # bytes → MB
        d_total_mem = d_props.total_memory / (1024**2)

        logger.info(
            f"Device [{i}]: name = [{d_name}], memory = [{d_total_mem / 1024:.1f} GB]")
        result[f'cuda:{i}'] = d_total_mem
    # end for

    return result


def _distribute_rodna_on_gpus(buffer_size: int = 512) -> Dict[str, int]:
    """Determines how many Rodna processes can be started
    on each GPU card."""

    # 1. See how much Rodna requires to load
    before_mem = torch.cuda.memory_allocated(device=_device)
    _ = RodnaProcessor(device=_device)
    after_mem = torch.cuda.memory_allocated(device=_device)
    # 1.1 Rounding up to the nearest power of 2 for total memory:
    # loaded + currently needed (for I/O tensors)
    needed_mem = after_mem - before_mem
    needed_mem /= 1024 ** 2
    needed_mem = needed_mem + buffer_size

    logger.info(f'Rodna needs [{needed_mem / 1024.0:.1f} GB] on the GPU card to load and run')

    # 2. See what machine we're running on
    machine_cuda_devices = _detect_cuda_devices()
    result = {}
    determined_processes = 0

    for d_name in machine_cuda_devices:
        d_memory = machine_cuda_devices[d_name]
        d_procs = d_memory // needed_mem
        # Make a bit of space, maybe some other apps need the GPU
        d_procs -= 1

        if d_procs == 0:
            logger.warning(f'Cannot load Rodna on GPU [{d_name}]: not enough memory')
        else:
            result[d_name] = int(d_procs)
            determined_processes += result[d_name]
        # end if
    # end for

    logger.info(f'Rodna can run in [{determined_processes}] processes, on all available GPU(s)')

    return result


def _count_words(file: str) -> int:
    with open(file, mode='r', encoding='utf-8') as f:
        text = f.read()
    # end with

    words = text.strip().split()
    return len(words)


def _rodna_worker(task_queue: mp.Queue, count_queue: mp.Queue, device: str):
    rodna = RodnaProcessor(device=torch.device(device=device))

    while True:
        file_path, wc = task_queue.get()

        if not file_path:
            break
        # end if

        rodna.process_text_file(txt_file=file_path)
        count_queue.put(wc)
    # end while

    logger.info(f'Process [{mp.current_process().name}] is done')


def _read_target_folder(folder: str) -> List[Tuple[str, int]]:
    result = []

    for f in os.listdir(path=folder):
        if f.endswith('.txt'):
            fp = os.path.join(folder, f)

            result.append((fp, _count_words(file=fp)))
        # end if
    # end for

    return result


def process_text_mp(folder: str):
    """Takes a `folder` with .txt files and processes them with the
    `RodnaProcessor`, using an auto-detected number of processes."""

    gpu_to_proc = _distribute_rodna_on_gpus()
    # Make sure we delete the test Rodna model from the GPU memory.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    if not gpu_to_proc:
        return
    # end if

    # Prepare tasks
    file_queue = mp.Queue()
    count_queue = mp.Queue()
    total_word_count = 0
    total_files_to_process = 0

    for fp, wc in _read_target_folder(folder=folder):
        file_queue.put((fp, wc))
        total_word_count += wc
        total_files_to_process += 1
    # end for

    # Prepare workers
    workers: List[mp.Process] = []
    p_count = 1

    for d_name in gpu_to_proc:
        for i in range(gpu_to_proc[d_name]):
            workers.append(mp.Process(
                target=_rodna_worker,
                args=(file_queue, count_queue, d_name)))
            workers[-1].name = f'Proc-Rodna-{p_count}'
            p_count += 1
        # end for
    # end for

    # Add 'end of task' messages for all processes
    for _ in range(p_count):
        file_queue.put((None, 0))
    # end for

    for w in workers:
        w.start()
    # end for

    with tqdm(total=total_word_count, unit="words", unit_scale=True) as pbar:
        pfc = 0

        while pfc < total_files_to_process:
            try:
                wc = count_queue.get(timeout=1., block=True)
                pfc += 1
                pbar.update(wc)
            except Empty:
                # Nothing is ready yet, wait a second more
                # and retry
                pass
            # end try
        # end while
    # end with

    # Wait for workers
    for w in workers:
        w.join(timeout=1.)

        if w.is_alive():
            logger.warning(
                f'Worker [{w.name}] has timed out. Killing it.')
            w.kill()
        # end if
    # end for



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python -m rodna.api.multiprocess <input folder with .txt files>',
              file=sys.stderr, flush=True)
        sys.exit(1)
    # end if

    input_path = sys.argv[1]

    if os.path.isdir(input_path):
        process_text_mp(folder=input_path)
    else:
        print('Usage: python -m rodna.api.multiprocess <input folder with .txt files>',
              file=sys.stderr, flush=True)
        sys.exit(1)
    # end if
