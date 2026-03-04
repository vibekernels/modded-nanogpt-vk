# modded-nanogpt-vk

A fork of [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) optimized for 1xH100.

## Running on RunPod

```bash
runpodctl pod create --name modded-nanogpt-vk --image ghcr.io/vibekernels/modded-nanogpt-vk:latest --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 1 --container-disk-in-gb 50 --ports "22/tcp"
```

## Training

Once the pod is running, SSH in and run:

```bash
cd /root/modded-nanogpt-vk
python data/cached_fineweb10B.py 9
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Performance History

| Commit | Description | Val Loss | Train Time |
|--------|-------------|----------|------------|
| [`dd11b65`](https://github.com/vibekernels/modded-nanogpt-vk/commit/dd11b6533423c707659feb14633bbfc3d505d4ff) | Initial fork | 3.2808 | 2294s |
| [`94639dc`](https://github.com/vibekernels/modded-nanogpt-vk/commit/94639dceceb440f032a760e75492817fab94c81d) | Move data prep type conversions and bigram hashing to GPU | 3.2802 | 668s |
