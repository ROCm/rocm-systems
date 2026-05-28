# rocDecode Test Scripts

## Pre-requisites to run python script
* Install [rocDecode](../../README.md#build-and-install-instructions)

* [FFMPEG](https://ffmpeg.org/about.html)

    * On `Ubuntu`

  ```shell
  sudo apt install libavcodec-dev libavformat-dev libavutil-dev
  ```
  
    * On `RHEL`/`SLES` - install ffmpeg development packages manually or use [rocDecode-setup.py](../../rocDecode-setup.py) script

* Python3 and pip packages - `pandas`, & ` tabulate`
  
```shell
python3 -m pip install pandas tabulate
```

## Scripts

**Usage:**

* **run_rocDecodeSamples.py**

```shell
usage: run_rocDecodeSamples.py [--rocDecode_directory ROCDECODE_DIRECTORY] 
                               [--gpu_device_id GPU_DEVICE_ID]
                               [--files_directory FILES_DIRECTORY]
                               [--sample_mode SAMPLE_MODE]
                               [--num_threads NUM_THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --rocDecode_directory ROCDECODE_DIRECTORY
                        The rocDecode Directory - required
  --gpu_device_id GPU_DEVICE_ID
                        The GPU device ID that will be used to run the test on it - optional (default:0 [range:0 - N-1] N = total number of available GPUs on a machine)
  --files_directory FILES_DIRECTORY
                        The path to a dirctory containing one or more supported files for decoding (e.g., mp4, mov, etc.) - required
  --sample_mode SAMPLE_MODE
                        The sample to run - optional (default:0 [range:0-1] 0: videoDecode, 1: videoDecodePerf)
  --num_threads NUM_THREADS
                        The number of threads is only for the videoDecodePerf sample (sample_mode = 1) - optional (default:1)
  --max_num_decoded_frames MAX_NUM_DECODED_FRAMES
                        The max number of decoded frames. Useful for partial decoding of a long stream. - optional (default:0, meaning no limit)
```

* **run_rocDecode_Conformance.py**

```shell
usage: run_rocDecode_Conformance.py [--rocDecode_directory ROCDECODE_DIRECTORY] 
                                    [--gpu_device_id GPU_DEVICE_ID]
                                    [--files_directory FILES_DIRECTORY]

optional arguments:
  -h, --help            show this help message and exit
  --rocDecode_directory ROCDECODE_DIRECTORY
                        The rocDecode Directory - required
  --gpu_device_id GPU_DEVICE_ID
                        The GPU device ID that will be used to run the test on it - optional (default:0 [range:0 - N-1] N = total number of available GPUs on a machine)
  --files_directory FILES_DIRECTORY
                        The path to a dirctory containing one or more supported files for decoding (e.g., mp4, mov, etc.) and their corresponding reference MD5 digests - required
```
