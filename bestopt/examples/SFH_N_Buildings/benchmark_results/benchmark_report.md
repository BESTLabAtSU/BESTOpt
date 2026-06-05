# BESTOpt Scaling Benchmark

_Generated: 2026-05-13T17:07:18.247679_

## System

- **platform**: linux
- **python_version**: 3.12.0
- **python_executable**: /home/zjiang19/miniconda3/envs/dynamic/bin/python
- **cpu_count_logical**: 128
- **cpu_count_physical**: 64
- **total_ram_gb**: 251.49
- **cpu_freq_mhz_current**: 1849.2381249999996
- **cpu_freq_mhz_max**: 2700.0
- **torch_version**: 2.10.0+cu128
- **cuda_available**: True
- **cuda_device_count**: 2
- **cuda_device_name**: NVIDIA RTX A6000
- **cuda_total_memory_gb**: 47.41

## Wall time per stage (seconds)

| N | status | config | env_init | sim | total | steps | avg step (ms) | steps/s |
|---|---|---|---|---|---|---|---|---|
| 1 | success | 0.366 | 1.713 | 8.051 | 10.13 | 192 | 41.934 | 23.847 |
| 5 | success | 0.005 | 2.518 | 39.298 | 41.821 | 192 | 204.675 | 4.886 |
| 10 | success | 0.007 | 4.903 | 78.38 | 83.29 | 192 | 408.227 | 2.45 |
| 30 | success | 0.018 | 14.713 | 235.073 | 249.804 | 192 | 1224.338 | 0.817 |
| 50 | success | 0.029 | 24.37 | 391.861 | 416.26 | 192 | 2040.94 | 0.49 |
| 100 | success | 0.055 | 54.036 | 776.097 | 830.188 | 192 | 4042.169 | 0.247 |
| 200 | success | 0.106 | 254.397 | 1532.429 | 1786.931 | 192 | 7981.4 | 0.125 |

## Memory (simulation stage)

| N | RAM proc max (MB) | RAM sys max (MB) | RAM sys max (%) | GPU mem max (MB) | GPU util max (%) |
|---|---|---|---|---|---|
| 1 | 1846.4 | 88273.0 | 35.2 | 8.1 | 0.0 |
| 5 | 5175.8 | 91659.7 | 36.5 | 8.2 | 0.0 |
| 10 | 9316.6 | 95928.4 | 38.2 | 8.2 | 0.0 |
| 30 | 25889.5 | 112428.3 | 44.6 | 8.4 | 0.0 |
| 50 | 42470.7 | 129140.8 | 51.1 | 16.6 | 0.0 |
| 100 | 83928.0 | 170579.5 | 67.2 | 9.1 | 0.0 |
| 200 | 166629.8 | 250992.7 | 98.9 | 18.0 | 0.0 |

## CPU (simulation stage)

| N | CPU proc mean (%) | CPU proc max (%) | CPU sys mean (%) | CPU sys max (%) |
|---|---|---|---|---|
| 1 | 197.68 | 1758.5 | 1.28 | 2.9 |
| 5 | 120.35 | 1692.3 | 1.1 | 2.8 |
| 10 | 110.48 | 1704.5 | 1.03 | 2.7 |
| 30 | 110.91 | 5101.4 | 1.07 | 22.1 |
| 50 | 102.05 | 1547.0 | 1.01 | 20.9 |
| 100 | 101.8 | 2567.5 | 0.96 | 23.8 |
| 200 | 101.02 | 2687.6 | 0.92 | 20.7 |
