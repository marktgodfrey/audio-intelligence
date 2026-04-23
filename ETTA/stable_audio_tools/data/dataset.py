# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.
# modified from stable-audio-tools under the MIT license

import importlib
import numpy as np
import io
import os
import posixpath
import random
import re
import shlex
import subprocess
import sys
import time
import torch
import torchaudio
import webdataset as wds
import math
from pathlib import Path
import json

from os import path
from torchaudio import transforms as T
from typing import Optional, Callable, List, Tuple
import soundfile as sf  # switch to soundfile for loading mp3
import librosa  # switch to librosa resampling

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T

from pytorch_lightning.utilities.rank_zero import rank_zero_only

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")
DEFAULT_S3_STREAMING_CONFIG = {
    "cli_connect_timeout_sec": 30,
    "cli_read_timeout_sec": 120,
    "ls_timeout_sec": 300,
    "stream_timeout_sec": 900,
    "stream_idle_timeout_sec": 300,
    "max_attempts": 3,
    "retry_mode": "standard",
}


def normalize_s3_streaming_config(overrides=None):
    config = dict(DEFAULT_S3_STREAMING_CONFIG)

    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})

    for key in (
        "cli_connect_timeout_sec",
        "cli_read_timeout_sec",
        "ls_timeout_sec",
        "stream_timeout_sec",
        "stream_idle_timeout_sec",
        "max_attempts",
    ):
        try:
            config[key] = int(config[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"s3_streaming.{key} must be an integer") from exc

        if config[key] <= 0:
            raise ValueError(f"s3_streaming.{key} must be greater than 0")

    retry_mode = str(config["retry_mode"]).strip()
    if not retry_mode:
        raise ValueError("s3_streaming.retry_mode must be a non-empty string")

    config["retry_mode"] = retry_mode
    return config


def build_aws_cli_base_cmd(s3_streaming_config, profile=None):
    config = normalize_s3_streaming_config(s3_streaming_config)
    cmd = [
        "aws",
        "--cli-connect-timeout",
        str(config["cli_connect_timeout_sec"]),
        "--cli-read-timeout",
        str(config["cli_read_timeout_sec"]),
    ]

    if profile is not None:
        cmd.extend(["--profile", profile])

    return cmd


def build_aws_cli_env(s3_streaming_config):
    config = normalize_s3_streaming_config(s3_streaming_config)
    env = os.environ.copy()
    env["AWS_MAX_ATTEMPTS"] = str(config["max_attempts"])
    env["AWS_RETRY_MODE"] = config["retry_mode"]
    return env


def build_s3_pipe_request(s3_path, profile=None, s3_streaming_config=None):
    config = normalize_s3_streaming_config(s3_streaming_config)
    s3_pipe_script = Path(__file__).with_name("s3_pipe.py")

    cmd = [
        shlex.quote(sys.executable),
        shlex.quote(str(s3_pipe_script)),
        "--s3-path",
        shlex.quote(s3_path),
        "--cli-connect-timeout-sec",
        str(config["cli_connect_timeout_sec"]),
        "--cli-read-timeout-sec",
        str(config["cli_read_timeout_sec"]),
        "--stream-timeout-sec",
        str(config["stream_timeout_sec"]),
        "--stream-idle-timeout-sec",
        str(config["stream_idle_timeout_sec"]),
        "--max-attempts",
        str(config["max_attempts"]),
        "--retry-mode",
        shlex.quote(config["retry_mode"]),
    ]

    if profile is not None:
        cmd.extend(["--profile", shlex.quote(profile)])

    return f"pipe:{' '.join(cmd)}"


def audio_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None


def collation_fn(samples):
    samples = [sample for sample in samples if sample is not None]
    if not samples:
        return None

    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.array(b)
        else:
            b = b
        result.append(b)
    return result


# Function to read NDJSON file and convert to a list of dictionaries. Used by compiling ndjson datasets to SA-compatible formats
def read_ndjson(file_path):
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]
    return data


# check if data is loadable with soundfile to filter out bad data
def check_file(filename, min_samplerate=44100):
    try:
        with sf.SoundFile(filename) as track:
            # Check for minimum frame count and sample rate
            bad_quality_criteria = (
                track.frames < 8192 or track.samplerate < min_samplerate
            )
            if bad_quality_criteria:
                print(f"filtering out bad quality filename {filename}")
                return False
            else:  # passed all criteria
                return True
    except Exception as e:
        print(f"Error thrown for {filename}: {e}")
        return False


# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py
def fast_scandir(
    min_samplerate: int,
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions,
    # max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = [
        "." + x if x[0] != "." else x for x in ext
    ]  # add starting period to extensions if needed
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if (
                        file_ext in ext
                        and not is_hidden
                        and check_file(f.path, min_samplerate)
                    ):  # new check_file condition
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(min_samplerate, dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def keyword_scandir(
    min_samplerate: int,
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ["." + x if x[0] != "." else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == "."
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any([keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words]
                    )
                    if (
                        has_ext
                        and has_keyword
                        and not has_banned
                        and not is_hidden
                        and not os.path.basename(f.path).startswith("._")
                    ):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(min_samplerate, dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


@rank_zero_only
def print_dataset_stats(dataset_stats: dict) -> None:
    # Define a template for each dataset's statistics with fixed field widths for headers and data
    header_template = "{:<30} {:>15} {:>15} {:>20}"
    data_template = "{:<30} {:>15d} {:>15d} {:>20.2f}"
    print("Dataset summary, sorted in descending #Samples order")
    print(
        header_template.format(
            "Dataset Name", "Samples", "Repeated", "#Samples/TotalSamples (%)"
        )
    )

    # Print each dataset's statistics
    for stat in dataset_stats:
        print(data_template.format(stat[0], stat[1], stat[2], stat[3]))


def get_audio_filenames(
    dataset_config_path: str,
    min_samplerate: int,
    dataset_ids: list,
    paths: list,  # directories in which to search
    n_repeats_per_dir: list,  # repeating the filenames to upsample certain dirs
    keywords=None,
    exts=[".wav", ".mp3", ".flac", ".ogg", ".aif", ".opus"],
):
    "recursively get a list of audio filenames"
    filenames = []
    num_files_per_dir = []

    if type(dataset_ids) is str:
        dataset_ids = [dataset_ids]
    if type(paths) is str:
        paths = [paths]
    if type(n_repeats_per_dir) is int:
        n_repeats_per_dir = [n_repeats_per_dir]

    filenames_cache_root = os.path.join(
        os.path.dirname(dataset_config_path), "filenames_cache"
    )
    if not os.path.exists(filenames_cache_root):
        os.makedirs(filenames_cache_root, exist_ok=True)

    for dataset_id, path, n_repeats in zip(
        dataset_ids, paths, n_repeats_per_dir
    ):  # get a list of relevant filenames
        filenames_cache_path = os.path.join(
            filenames_cache_root,
            os.path.basename(dataset_config_path) + "_" + dataset_id + ".txt",
        )
        if os.path.exists(filenames_cache_path):
            print(f"using filenames cache from {filenames_cache_path}")
            with open(filenames_cache_path, "r") as f:
                files = f.read().splitlines()
        else:
            print(
                f"gathering and checking audio filenames with sample rate >= {min_samplerate} from: {path}"
            )
            if keywords is not None:
                subfolders, files = keyword_scandir(
                    min_samplerate, path, exts, keywords
                )
            else:
                subfolders, files = fast_scandir(min_samplerate, path, exts)
            print(f"saving filenames cache to {filenames_cache_path}")
            with open(filenames_cache_path, "w") as f:
                for filename in files:
                    f.write(filename + "\n")

        print(f"{path} has {len(files)} samples, repeated {n_repeats}x")
        files = files * (n_repeats + 1)
        filenames.extend(files)
        num_files_per_dir.append(len(files))  # repeated

    return filenames, num_files_per_dir


def get_latent_filenames(paths: list, extensions=["npy"]):
    """recursively get a list of pre-encoded filenames"""
    filenames = []

    if type(paths) is str:
        paths = [paths]

    # normalize extensions to include leading dot
    extensions = [
        "." + ext.lower().lstrip(".")
        for ext in extensions
    ]

    for path in paths:
        filelist_path = os.path.join(path, "filelist.txt")
        if os.path.exists(filelist_path):
            with open(filelist_path, "r") as f:
                files = [os.path.join(path, file.strip()) for file in f.readlines()]
            filenames.extend(files)
            continue

        for root, _, files in os.walk(path):
            for fname in files:
                if fname.startswith(".") or fname.startswith("._"):
                    continue
                if os.path.splitext(fname)[1].lower() in extensions:
                    filenames.append(os.path.join(root, fname))

    return filenames


class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        n_repeats: int,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        metadata: Optional[
            dict
        ] = None,  # any other custom k-v dict that holds metadata from this dataset
    ):
        self.id = id
        self.path = path
        self.n_repeats = n_repeats
        self.custom_metadata_fn = custom_metadata_fn
        self.metadata = metadata


class LocalWebDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):
        _, self.urls = fast_scandir(0, self.path, ["tar"])
        return self.urls


class S3DatasetConfig:
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
        s3_streaming_config: Optional[dict] = None,
    ):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.s3_streaming_config = normalize_s3_streaming_config(s3_streaming_config)
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
            s3_streaming_configs={self.path: self.s3_streaming_config},
        )
        return self.urls


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs,
        dataset_config_path,
        dataset_type,
        sample_size=65536,
        sample_rate=48000,
        keywords=None,
        random_crop=True,
        force_channels="stereo",
        print_stats=True,
        custom_metadata_fn: Optional[
            Callable[[str], str]
        ] = None,  # "global" metadata_fn used by dataset_type other than "audio_dir"
    ):
        super().__init__()
        self.filenames = []
        self.dataset_stats = []
        self.metadata = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )

        self.root_paths = []

        self.sample_size = sample_size
        self.random_crop = random_crop

        self.pad_crop = PadCrop_Normalized_T(
            sample_size, sample_rate, randomize=random_crop
        )

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate

        self.custom_metadata_fns = {}

        self.dataset_type = dataset_type

        if self.dataset_type == "audio_dir":  # original stable-audio-tools
            for config in configs:
                self.root_paths.append(config.path)
                filenames, num_files_per_dir = get_audio_filenames(
                    dataset_config_path=dataset_config_path,
                    min_samplerate=self.sr,
                    dataset_ids=config.id,
                    paths=config.path,
                    n_repeats_per_dir=config.n_repeats,
                    keywords=keywords,
                )
                self.filenames.extend(filenames)
                # NOTE: metadata is not used by audio_dir
                self.metadata.extend(
                    [config.metadata for _ in range(len(filenames))]
                )  # dummy metadata matching filenames length
                self.dataset_stats.append(
                    (config.id, num_files_per_dir, config.n_repeats)
                )
                if config.custom_metadata_fn is not None:
                    self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        elif self.dataset_type == "location_caption_pair":
            assert (
                custom_metadata_fn is not None
            ), f"for dataset_type {self.dataset_type}, provide global custom_metadata_fn to SampleDataset!"
            for config in configs:
                self.filenames.append(config.path)
                self.metadata.append(
                    config.metadata
                )  # treat all k-v dict per data point as this config's metadata
            self.root_paths = {os.path.dirname(path) for path in self.filenames}
            for root_path in self.root_paths:
                self.custom_metadata_fns[root_path] = (
                    custom_metadata_fn  # assumes same fn is used for all root_paths
                )

        else:
            raise ValueError(f"unknown dataset_type {self.dataset_type}")

        print(f"Found {len(self.filenames)} files")

        if print_stats:
            assert (
                self.dataset_type == "audio_dir"
            ), f"currently print_stats is not supported for dataset_type {self.dataset_type}"
            updated_dataset_stats = []
            for dataset_id, num_files_per_dir, repeats in self.dataset_stats:
                for num_files in num_files_per_dir:
                    effective_sample_share = (num_files / len(self.filenames)) * 100
                    updated_dataset_stats.append(
                        (dataset_id, num_files, repeats, effective_sample_share)
                    )
            self.dataset_stats = updated_dataset_stats
            # Sort datasets by the number of samples in descending order
            self.dataset_stats.sort(key=lambda x: x[1], reverse=True)
            print_dataset_stats(self.dataset_stats)

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        if ext == "mp3":
            # new impl using SoundFile
            track = sf.SoundFile(filename)
            if not track.seekable():  # corner case, re-roll
                raise ValueError(f"audio is not seekable: {filename}")
            if track.frames == 0:  # corner case, re-roll
                raise ValueError(f"audio has zero frames: {filename}")
            audio = track.read(track.frames, always_2d=True)
            audio = torch.permute(
                torch.from_numpy(audio), (1, 0)
            ).float()  # soundfile has shape [time, channel] so permute it back
            in_sr = track.samplerate
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)  # [channel, time]

        # sanity check assertions to ensure that the audio contains the singal
        if audio.numel() < 8192:
            raise ValueError(f"audio_numel() < 8192: {filename}")

        if abs(audio).max() < 1e-4:
            raise ValueError(
                f"entire audio is silence abs(audio).max() < 1e-4: {filename}"
            )

        if len(audio.shape) != 2:
            raise ValueError(f"audio.shape {audio.shape}: {filename}")

        if audio.shape[0] > audio.shape[1]:  # audio is not in shape of [channel, time]
            raise ValueError(
                f"audio.shape {audio.shape}, did you forget to permute it to [channel, time]?: {filename}"
            )

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    # new load_file implementation with cleaner code and soundfile seeking
    # plus the original logic of PadCrop_Normalized_T to reflect stable audio's usage of timestamp conditioning when loading chunk from seeking
    def load_file_and_pad_crop(
        self,
        filename: str,
        sample_size: int,
        target_sr: int = 44100,
        start_sec: float = 0.0,
        end_sec: float = 3600.0,
        randomize: bool = True,
    ) -> Tuple[torch.Tensor, float, float, int, int]:

        # soundfile version
        track = sf.SoundFile(filename)
        if not track.seekable():  # corner case, re-roll
            raise ValueError(f"audio is not seekable: {filename}")
        if track.frames == 0:  # corner case, re-roll
            raise ValueError(f"audio has zero frames: {filename}")
        original_sr = track.samplerate
        original_channels = track.channels
        num_frames_total = track.frames
        seconds_total = num_frames_total / original_sr

        if Path(filename).suffix.lower() != ".wav" and seconds_total > 7200:
            raise ValueError(
                f"non-wav audio file is too long ({seconds_total} seconds > 7200, 2 hours) which will impact training throughput. Skipping: {filename}"
            )

        if randomize:  # random chunk loading for given sample_size
            target_sample_size = sample_size
            if (
                original_sr != target_sr
            ):  # adjust sample size to load if the source sr is different
                target_sample_size = math.ceil(
                    target_sample_size * (original_sr / target_sr)
                )
            upper_bound = max(0, num_frames_total - target_sample_size)
            num_frames_to_skip = random.randint(0, upper_bound)
        else:  # use given start_sec and end_sec
            num_frames_to_skip = int(start_sec * original_sr)
            target_sample_size = int(end_sec * original_sr) - num_frames_to_skip

        # soundfile version
        track.seek(num_frames_to_skip)
        num_frames_to_read = min(
            target_sample_size, num_frames_total - num_frames_to_skip
        )
        audio = track.read(num_frames_to_read, always_2d=True)

        # sanity check assertions to ensure that the audio contains valid singal. Do the checks before maybe costly resampling
        if audio.size < 8192:
            raise ValueError(f"audio.size < 8192: {filename}")

        if abs(audio).max() < 1e-4:
            raise ValueError(f"audio is silence abs(audio).max() < 1e-4: {filename}")

        if len(audio.shape) != 2:
            raise ValueError(f"audio.shape {audio.shape}: {filename}")

        if audio.shape[0] < audio.shape[1]:  # audio is not in shape of [time, channel]
            raise ValueError(
                f"audio.shape {audio.shape}, track.read should have returned [time, channel]: {filename}"
            )

        # retrieve spectral rolloff to get a "true" frequency this audio has
        # our dataset contains force-upsampled audio data that may not contain valid frequency information in high band
        spectral_rolloff = int(
            librosa.feature.spectral_rolloff(
                y=audio.mean(-1), sr=original_sr, roll_percent=0.999
            ).mean()
        )

        # soundfile has shape [time, channel] so permute it back
        audio = audio.T  # [channel, time]

        if original_sr != target_sr:
            audio = librosa.resample(
                audio, orig_sr=original_sr, target_sr=target_sr, res_type="soxr_vhq"
            )  # change from soxr_hq to better soxr_vhq, no impact in speed

        # cast to torch tensor
        audio = torch.from_numpy(audio).float()

        if audio.shape[-1] > sample_size:
            # trim last element to match sample_size (e.g., 16385 for 44khz downsampled to 24khz -> 16384)
            audio = audio[..., :sample_size]

        # compute additional metadata used by stable audio models.
        # this uses original full clip (num_frames_total) to refelect original intention of timestamp embeddings
        t_start = num_frames_to_skip / num_frames_total
        t_end = (num_frames_to_skip + num_frames_to_read) / num_frames_total

        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(num_frames_to_skip / original_sr)
        seconds_total = math.ceil(seconds_total)

        # Create the chunk
        num_channels, num_audio_samples = audio.shape
        chunk = audio.new_zeros([num_channels, sample_size])
        chunk[:, :num_audio_samples] = audio[:, : min(sample_size, num_audio_samples)]

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([sample_size])
        padding_mask[:num_audio_samples] = 1

        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask,
            original_sr,
            original_channels,
            spectral_rolloff,
        )

    def __len__(self):
        return len(self.filenames)

    def load_audio_and_info(self, idx):
        audio_filename = self.filenames[idx]
        metadata = self.metadata[idx]

        start_time = time.time()

        # set audio_start_sec and audio_end_sec if defined in metadata.
        # If the "start" and "end" keys are not found or metadata itself is not defined (None), auto-set to load full audio file
        audio_start_sec = (metadata or {}).get("start", 0.0)
        audio_end_sec = (metadata or {}).get(
            "end", 3600.0
        )  # 1 hour (arbitrary large value, will be auto-adjusted to end_sec of actual file from loading)

        (
            audio,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask,
            original_sr,
            original_channels,
            spectral_rolloff,
        ) = self.load_file_and_pad_crop(
            filename=audio_filename,
            sample_size=self.sample_size,
            target_sr=self.sr,
            start_sec=audio_start_sec,
            end_sec=audio_end_sec,
            randomize=self.random_crop,
        )

        # Run augmentations on this sample (including random crop)
        if self.augs is not None:
            audio = self.augs(audio)

        audio = audio.clamp(-1, 1)

        # Encode the file to assist in prediction
        if self.encoding is not None:
            audio = self.encoding(audio)

        info = {}

        info["path"] = audio_filename

        for root_path in self.root_paths:
            if root_path in audio_filename:
                info["relpath"] = path.relpath(audio_filename, root_path)
                break

        info["timestamps"] = (t_start, t_end)
        info["seconds_start"] = seconds_start
        info["seconds_total"] = seconds_total
        info["padding_mask"] = padding_mask
        info["original_sr"] = original_sr
        info["original_channels"] = original_channels
        info["spectral_rolloff"] = spectral_rolloff

        # new addition that stores any k-v metadata
        info["metadata"] = metadata

        end_time = time.time()

        info["load_time"] = end_time - start_time

        return (audio, info)

    def __getitem__(self, idx):
        try:
            # load audio and info
            audio, info = self.load_audio_and_info(idx)

            # augment audio and info based on custom_metadata_fn
            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in self.filenames[idx]:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    # self (dataset) is also the input to the custom_metadata_fn for flexible data augmentation
                    # info is updated with potential new k-v based on the logic of custom_metadata_fn
                    audio, info = custom_metadata_fn(self, audio, info)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

            return (audio, info)

        except Exception as e:
            print(
                f"[WARNING(SampleDataset.__getitem__)] Couldn't load file {self.filenames[idx]}: {e}"
            )
            return self[random.randrange(len(self))]


class PreEncodedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            configs: List[LocalDatasetConfig],
            latent_crop_length=None,
            min_length_sec=None,
            max_length_sec=None,
            random_crop=False,
            latent_extension='npy'
    ):
        super().__init__()
        self.filenames = []

        self.custom_metadata_fns = {}

        self.latent_extension = latent_extension

        for config in configs:
            self.filenames.extend(get_latent_filenames(config.path, [latent_extension]))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop

        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        print(f'Found {len(self.filenames)} files')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        latent_filename = self.filenames[idx]
        try:
            latents = torch.from_numpy(np.load(latent_filename))  # [C, N]

            md_filename = latent_filename.replace(f".{self.latent_extension}", ".json")

            with open(md_filename, "r") as f:
                try:
                    info = json.load(f)
                except:
                    raise Exception(f"Couldn't load metadata file {md_filename}")

            info["latent_filename"] = latent_filename

            if self.latent_crop_length is not None:

                # Get the last index from the padding mask, the index of the last 1 in the sequence
                last_ix = len(info["padding_mask"]) - 1 - info["padding_mask"][::-1].index(1)

                if self.random_crop and last_ix > self.latent_crop_length:
                    start = random.randint(0, last_ix - self.latent_crop_length)
                else:
                    start = 0

                latents = latents[:, start:start + self.latent_crop_length]

                info["padding_mask"] = info["padding_mask"][start:start + self.latent_crop_length]

                info["latent_crop_length"] = self.latent_crop_length
                info["latent_crop_start"] = start

            info["padding_mask"] = [torch.tensor(info["padding_mask"])]

            seconds_total = info["seconds_total"]

            if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                return self[random.randrange(len(self))]

            if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                return self[random.randrange(len(self))]

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in latent_filename:
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, None)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                if "__replace__" in info and info["__replace__"] is not None:
                    # Replace the latents with the new latents if the custom metadata function returns a new set of latents
                    latents = info["__replace__"]

            info["audio"] = latents

            return (latents, info)
        except Exception as e:
            print(f'Couldn\'t load file {latent_filename}: {e}')
            return self[random.randrange(len(self))]


def get_dbmax(audio):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20 * torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()


def is_silence(audio, thresh=-60):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh


def get_s3_contents(
    dataset_path,
    s3_url_prefix=None,
    filter="",
    recursive=True,
    debug=False,
    profile=None,
    s3_streaming_config=None,
):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    s3_streaming_config = normalize_s3_streaming_config(s3_streaming_config)

    if dataset_path != "" and not dataset_path.endswith("/"):
        dataset_path += "/"

    bucket_path = posixpath.join(s3_url_prefix or "", dataset_path)
    cmd = build_aws_cli_base_cmd(s3_streaming_config, profile=profile)
    cmd.extend(["s3", "ls", bucket_path])

    if recursive:
        cmd.append("--recursive")

    run_ls = subprocess.run(
        cmd,
        capture_output=True,
        check=True,
        timeout=s3_streaming_config["ls_timeout_sec"],
        env=build_aws_cli_env(s3_streaming_config),
    )
    contents = run_ls.stdout.decode("utf-8").split("\n")
    contents = [x.strip() for x in contents if x]
    contents = [
        re.sub(r"^\S+\s+\S+\s+\d+\s+", "", x)
        if re.match(r"^\S+\s+\S+\s+\d+\s+", x)
        else x
        for x in contents
    ]
    contents = [
        posixpath.join(s3_url_prefix or "", x) for x in contents if not x.endswith("/")
    ]

    if filter:
        contents = [x for x in contents if filter in x]

    if recursive:
        main_dir = "/".join(bucket_path.split("/")[3:])
        contents = [x.replace(f"{main_dir}", "").replace("//", "/") for x in contents]

    if debug:
        print("contents = \n", contents)

    return contents


def get_all_s3_urls(
    names=None,
    subsets=None,
    s3_url_prefix=None,
    recursive=True,
    filter_str="tar",
    debug=False,
    profiles=None,
    s3_streaming_configs=None,
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    names = names or []
    subsets = subsets or [""]
    profiles = profiles or {}
    s3_streaming_configs = s3_streaming_configs or {}
    urls = []

    for name in names:
        if s3_url_prefix is None:
            contents_str = name
        else:
            contents_str = posixpath.join(s3_url_prefix, name)

        if debug:
            print(f"get_all_s3_urls: {contents_str}:")

        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")

            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str,
                s3_url_prefix=None,
                recursive=recursive,
                filter=filter_str,
                debug=debug,
                profile=profile,
                s3_streaming_config=s3_streaming_configs.get(name),
            )

            for tar in tar_list:
                s3_path = posixpath.join(name, subset, tar)

                if s3_url_prefix is None:
                    request_str = build_s3_pipe_request(
                        s3_path,
                        profile=profiles.get(name),
                        s3_streaming_config=s3_streaming_configs.get(name),
                    )
                else:
                    request_str = build_s3_pipe_request(
                        posixpath.join(s3_url_prefix, s3_path),
                        profile=profiles.get(name),
                        s3_streaming_config=s3_streaming_configs.get(name),
                    )

                if debug:
                    print("request_str = ", request_str)

                urls.append(request_str)

    return urls


def log_and_continue(exn):
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def is_valid_sample(sample):
    if sample is None or not isinstance(sample, dict):
        return False

    json_data = sample.get("json")
    audio_data = sample.get("audio")
    has_json = isinstance(json_data, dict)
    has_audio = audio_data is not None
    is_pre_encoded = sample.get("__pre_encoded__", False)
    is_silent = has_audio and (not is_pre_encoded) and is_silence(audio_data)
    is_rejected = has_json and json_data.get("__reject__", False)

    return has_json and has_audio and not is_silent and not is_rejected


def has_valid_prompt(metadata):
    if "prompt" not in metadata:
        return False

    prompt = metadata["prompt"]

    if prompt is None:
        return False

    if isinstance(prompt, str):
        return bool(prompt.strip())

    return True


class WebDatasetDataLoader:
    def __init__(
        self,
        datasets: List[LocalWebDatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        pre_encoded=False,
        latent_crop_length=None,
        latent_extension="npy",
        min_length_sec=None,
        max_length_sec=None,
        resampled_shards=True,
        **data_loader_kwargs,
    ):
        self.datasets = datasets
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase
        self.pre_encoded = pre_encoded
        self.latent_crop_length = latent_crop_length
        self.latent_extension = latent_extension.lower().lstrip(".")
        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        urls = [dataset.load_data_urls() for dataset in datasets]
        urls = [url for dataset_urls in urls for url in dataset_urls]
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls) if resampled_shards else wds.SimpleShardList(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(self._decoder(), handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        )

        if resampled_shards:
            self.dataset = self.dataset.with_epoch(
                epoch_steps // num_workers if num_workers > 0 else epoch_steps
            )

        def worker_init_fn(worker_id):
            torch.multiprocessing.set_sharing_strategy("file_system")

        self.data_loader = wds.WebLoader(
            self.dataset,
            batch_size=None,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            **data_loader_kwargs,
        )

    def _decoder(self):
        if not self.pre_encoded:
            return audio_decoder

        latent_extension = self.latent_extension

        def latent_decoder(key, value):
            ext = key.split(".")[-1].lower()
            if ext == latent_extension:
                return np.lib.format.read_array(io.BytesIO(value))
            return None

        return latent_decoder

    def _apply_pre_encoded_rules(self, latents, info, source_path, custom_metadata_fn):
        info = dict(info)
        info["latent_filename"] = source_path

        if self.latent_crop_length is not None:
            last_ix = len(info["padding_mask"]) - 1 - info["padding_mask"][::-1].index(1)

            if self.random_crop and last_ix > self.latent_crop_length:
                start = random.randint(0, last_ix - self.latent_crop_length)
            else:
                start = 0

            latents = latents[:, start : start + self.latent_crop_length]
            info["padding_mask"] = info["padding_mask"][start : start + self.latent_crop_length]
            info["latent_crop_length"] = self.latent_crop_length
            info["latent_crop_start"] = start

        info["padding_mask"] = [torch.tensor(info["padding_mask"])]

        seconds_total = info["seconds_total"]

        if self.min_length_sec is not None and seconds_total < self.min_length_sec:
            info["__reject__"] = True

        if self.max_length_sec is not None and seconds_total > self.max_length_sec:
            info["__reject__"] = True

        if custom_metadata_fn is not None:
            custom_metadata = custom_metadata_fn(info, None)
            info.update(custom_metadata)

            if "__replace__" in info and info["__replace__"] is not None:
                latents = info["__replace__"]

        info["audio"] = latents

        return latents, info

    def wds_preprocess(self, sample):
        metadata = sample.get("json")
        if not isinstance(metadata, dict):
            return None

        if self.pre_encoded:
            audio = None
            found_key = self.latent_extension

            for key in list(sample.keys()):
                if key.endswith(self.latent_extension):
                    found_key = key
                    break

            if found_key not in sample:
                return None

            audio = torch.from_numpy(sample[found_key])
            del sample[found_key]
            sample["__pre_encoded__"] = True
        else:
            found_key = ""
            for key in sample.keys():
                for akey in AUDIO_KEYS:
                    if key.endswith(akey):
                        found_key = key
                        break
                if found_key:
                    break

            if not found_key:
                return None

            audio, in_sr = sample[found_key]
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate)
                audio = resample_tf(audio)

            if self.sample_size is not None:
                pad_crop = PadCrop_Normalized_T(
                    self.sample_size,
                    randomize=self.random_crop,
                    sample_rate=self.sample_rate,
                )
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(audio)
                metadata["seconds_start"] = seconds_start
                metadata["seconds_total"] = seconds_total
                metadata["padding_mask"] = padding_mask
            else:
                t_start, t_end = 0, 1

            if audio.shape[-1] == 0:
                audio = torch.zeros(1, 1)

            augs = torch.nn.Sequential(
                Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
                Mono() if self.force_channels == "mono" else torch.nn.Identity(),
                PhaseFlipper() if self.augment_phase else torch.nn.Identity(),
            )
            audio = augs(audio)
            metadata["timestamps"] = (t_start, t_end)
            metadata["audio"] = audio

        if "text" in metadata:
            metadata["prompt"] = metadata["text"]

        matched_dataset = None
        for dataset in self.datasets:
            if dataset.path not in sample["__url__"]:
                continue

            matched_dataset = dataset

            if self.pre_encoded:
                audio, info = self._apply_pre_encoded_rules(
                    audio,
                    metadata,
                    sample["__key__"],
                    dataset.custom_metadata_fn,
                )
                metadata = info
                sample["json"] = info
                break
            elif dataset.custom_metadata_fn is not None:
                custom_metadata = dataset.custom_metadata_fn(metadata, audio)
                metadata.update(custom_metadata)

        if self.pre_encoded and matched_dataset is None:
            audio, info = self._apply_pre_encoded_rules(
                audio,
                metadata,
                sample["__key__"],
                None,
            )
            metadata = info
            sample["json"] = info

        if not has_valid_prompt(metadata):
            metadata["__reject__"] = True
            metadata["__reject_reason__"] = "missing_prompt"

        sample["audio"] = audio
        metadata["audio"] = audio

        return sample


def create_webdataset_configs(dataset_entries, dataset_type, s3_streaming_defaults=None):
    configs = []

    for wds_config in dataset_entries:
        custom_metadata_fn = None
        custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location(
                "metadata_module", custom_metadata_module_path
            )
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)
            custom_metadata_fn = metadata_module.get_custom_metadata

        if dataset_type == "wds":
            webdataset_path = wds_config.get("path", None)
            assert webdataset_path is not None, "Path must be set for local WebDataset configuration"

            configs.append(
                LocalWebDatasetConfig(
                    id=wds_config["id"],
                    path=webdataset_path,
                    custom_metadata_fn=custom_metadata_fn,
                )
            )
        else:
            s3_path = wds_config.get("s3_path", None)
            assert s3_path is not None, "s3_path must be set for S3 WebDataset configuration"

            s3_streaming_config = dict(s3_streaming_defaults or {})
            s3_streaming_config.update(wds_config.get("s3_streaming", {}))

            configs.append(
                S3DatasetConfig(
                    id=wds_config["id"],
                    s3_path=s3_path,
                    custom_metadata_fn=custom_metadata_fn,
                    profile=wds_config.get("profile", None),
                    s3_streaming_config=s3_streaming_config,
                )
            )

    return configs

def group_by_keys(
    data, keys=wds.tariterators.base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        if not isinstance(filesample, dict):
            continue

        fname = filesample.get("fname", None)
        value = filesample.get("data", None)

        # WebDataset can emit control/error records that are not actual tar members.
        # Skip them rather than crashing worker processes.
        if fname is None or value is None:
            continue

        prefix, suffix = keys(fname)
        if wds.tariterators.trace:
            print(
                prefix,
                suffix,
                current_sample.keys() if isinstance(current_sample, dict) else None,
            )
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if wds.tariterators.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(
                __key__=prefix,
                __url__=filesample.get("__url__", ""),
            )
        if suffix in current_sample:
            print(
                f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}"
            )
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if wds.tariterators.valid_sample(current_sample):
        yield current_sample


wds.tariterators.group_by_keys = group_by_keys


def create_dataloader_from_config(
    dataset_config,
    dataset_config_path,
    batch_size,
    sample_size,
    sample_rate,
    audio_channels=2,
    num_workers=4,
):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = "stereo"

    # List of multiple Location caption pair manifest json files.
    # Convert this to "location_caption_pair" dataset_type below by aggregating muptiple json manifests into one
    if dataset_type == "location_caption_pair_manifests":
        print(
            f"[INFO] dataset_type is location_caption_pair_manifests. Will load each manifest files and compile them into location_caption_pair dataset_type format!"
        )
        # Read the master JSON file that contains multiple manifest json file paths
        with open(dataset_config_path, "r") as f:
            manifests_data = json.load(f)
        # Initialize an empty list to hold all dataset entries
        all_datasets = []
        all_datasets_valid = []

        # Read each NDJSON file and append its data to the all_datasets list
        for dataset in manifests_data["datasets"]:
            ndjson_path = dataset["path"]
            # print(f"[INFO] reading {ndjson_path}")
            ndjson_data = read_ndjson(ndjson_path)
            all_datasets.extend(ndjson_data)
        for dataset in manifests_data["datasets_valid"]:
            ndjson_path = dataset["path"]
            # print(f"[INFO] reading {ndjson_path}")
            ndjson_data = read_ndjson(ndjson_path)
            # note that it appends to consider each data as separate validation datasets to track
            all_datasets_valid.append(ndjson_data)

        # Convert to dict that follows location_caption_pair format
        dataset_config = {
            "dataset_type": "location_caption_pair",
            "custom_metadata_module": manifests_data.get("custom_metadata_module", ""),
            "custom_metadata_module_valid": manifests_data.get(
                "custom_metadata_module_valid", ""
            ),
            "random_crop": manifests_data.get("random_crop", False),
            "datasets": all_datasets,
            "datasets_valid": all_datasets_valid,
        }
        # override dataset_type to location_caption_pair to trigger the dataset loading logic below
        dataset_type = "location_caption_pair"

    # Location caption pair used by af. Convert json manifest to stable-audio-tools dataloader format
    if dataset_type == "location_caption_pair":
        datasets = dataset_config.get("datasets", None)
        assert (
            datasets is not None
        ), "location caption pair must be specified under the key 'datasets'"

        # custom_metadata_module will be defined to the root, not per-sample level
        custom_metadata_fn = None
        custom_metadata_module_path = dataset_config.get("custom_metadata_module", None)
        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location(
                "metadata_module", custom_metadata_module_path
            )
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)
            custom_metadata_fn = metadata_module.get_custom_metadata

        # create training data configs
        configs = []
        for location_caption_pair in datasets:
            audio_dir_path = location_caption_pair.get("location", None)
            assert (
                audio_dir_path is not None
            ), "Path must be set for local audio directory configuration"
            configs.append(
                LocalDatasetConfig(
                    id=location_caption_pair["dataset"],
                    path=audio_dir_path,
                    n_repeats=location_caption_pair.get("n_repeats", 0),
                    # custom_metadata_fn=custom_metadata_fn,
                    metadata=location_caption_pair,
                )
            )

        # create train set
        train_set = SampleDataset(
            configs,
            dataset_config_path,
            dataset_type,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get(
                "random_crop", False
            ),  # defaults to false for TTA training
            force_channels=force_channels,
            print_stats=False,  # each config points to audio file so print_stats is not defined
            custom_metadata_fn=custom_metadata_fn,
        )

        # create train data loader
        train_dl = torch.utils.data.DataLoader(
            train_set,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=collation_fn,
        )

        # create validation data configs and valid sets. Note that datasets_valid is list of list
        datasets_valid = dataset_config.get("datasets_valid", None)
        list_valid_dl = []

        if datasets_valid is None:
            print(
                f"[WARNING (dataset.py)]: datasets_valid not found. This will disable tracking losses from validation sets"
            )

        else:
            # custom_metadata_module_valid will be defined to the root, not per-sample level
            custom_metadata_fn_valid = None
            custom_metadata_module_path_valid = dataset_config.get(
                "custom_metadata_module_valid", None
            )
            if custom_metadata_module_path_valid is not None:
                spec = importlib.util.spec_from_file_location(
                    "metadata_module", custom_metadata_module_path_valid
                )
                metadata_module_valid = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module_valid)
                custom_metadata_fn_valid = metadata_module_valid.get_custom_metadata

            for dataset_valid in datasets_valid:
                configs_valid = []
                for location_caption_pair in dataset_valid:
                    audio_dir_path = location_caption_pair.get("location", None)
                    assert (
                        audio_dir_path is not None
                    ), "Path must be set for local audio directory configuration"
                    configs_valid.append(
                        LocalDatasetConfig(
                            id=location_caption_pair["dataset"],
                            path=audio_dir_path,
                            n_repeats=0,
                            # custom_metadata_fn=custom_metadata_fn_valid,
                            metadata=location_caption_pair,
                        )
                    )
                valid_set = SampleDataset(
                    configs_valid,
                    dataset_config_path,
                    dataset_type,
                    sample_rate=sample_rate,
                    sample_size=sample_size,  # follows the same as training, so if it's 10s it'll trim validation audio to 10s to keep that in mind
                    random_crop=False,  # defaults to false for TTA evaluation
                    force_channels=force_channels,
                    print_stats=False,  # each config points to audio file so print_stats is not defined
                    custom_metadata_fn=custom_metadata_fn_valid,
                )
                valid_dl = torch.utils.data.DataLoader(
                    valid_set,
                    batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    persistent_workers=False,
                    pin_memory=False,
                    drop_last=False,
                    collate_fn=collation_fn,
                )
                list_valid_dl.append(valid_dl)

        return train_dl, list_valid_dl

    # original stable-audio-tools dataset_type. Used by autoencoder training that scans all audio files of given dirs
    elif dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)
        assert (
            audio_dir_configs is not None
        ), 'Directory configuration must be specified in datasets["dataset"]'

        configs = []
        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert (
                audio_dir_path is not None
            ), "Path must be set for local audio directory configuration"

            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get(
                "custom_metadata_module", None
            )

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location(
                    "metadata_module", custom_metadata_module_path
                )
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)
                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    n_repeats=audio_dir_config.get("n_repeats", 0),
                    custom_metadata_fn=custom_metadata_fn,
                )
            )

        train_set = SampleDataset(
            configs,
            dataset_config_path,
            dataset_type,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get(
                "random_crop", True
            ),  # defaults to true for autoencoder training
            force_channels=force_channels,
        )

        train_dl = torch.utils.data.DataLoader(
            train_set,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=False,
            pin_memory=True,
            drop_last=True,
            collate_fn=collation_fn,
        )

        list_valid_dl = []  # assumes no validation dataloaders for now

        return train_dl, list_valid_dl

    elif dataset_type == "pre_encoded":

        pre_encoded_dir_configs = dataset_config.get("datasets", None)

        assert pre_encoded_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        latent_crop_length = dataset_config.get("latent_crop_length", None)
        min_length_sec = dataset_config.get("min_length_sec", None)
        max_length_sec = dataset_config.get("max_length_sec", None)
        random_crop = dataset_config.get("random_crop", False)
        latent_extension = dataset_config.get("latent_extension", 'npy')

        configs = []

        for pre_encoded_dir_config in pre_encoded_dir_configs:
            pre_encoded_dir_path = pre_encoded_dir_config.get("path", None)
            assert pre_encoded_dir_path is not None, "Path must be set for local audio directory configuration"

            custom_metadata_fn = None
            custom_metadata_module_path = pre_encoded_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)

                custom_metadata_fn = metadata_module.get_custom_metadata

            configs.append(
                LocalDatasetConfig(
                    id=pre_encoded_dir_config["id"],
                    path=pre_encoded_dir_path,
                    n_repeats=0,
                    custom_metadata_fn=custom_metadata_fn
                )
            )

        train_set = PreEncodedDataset(
            configs,
            latent_crop_length=latent_crop_length,
            min_length_sec=min_length_sec,
            max_length_sec=max_length_sec,
            random_crop=random_crop,
            latent_extension=latent_extension
        )

        train_dl = torch.utils.data.DataLoader(
            train_set,
            batch_size,
            shuffle=dataset_config.get("shuffle", True),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=dataset_config.get("drop_last", True),
            collate_fn=collation_fn,
        )

        # create validation data configs and valid sets. Note that datasets_valid is list of list
        pre_encoded_dir_configs_valid = dataset_config.get("datasets_valid", None)
        list_valid_dl = []

        if pre_encoded_dir_configs_valid is not None:
            configs_valid = []

            for pre_encoded_dir_config_valid in pre_encoded_dir_configs_valid:
                pre_encoded_dir_path = pre_encoded_dir_config_valid.get("path", None)
                assert pre_encoded_dir_path is not None, "Path must be set for local audio directory configuration"

                custom_metadata_fn = None
                custom_metadata_module_path = pre_encoded_dir_config_valid.get("custom_metadata_module", None)

                if custom_metadata_module_path is not None:
                    spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                    metadata_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(metadata_module)

                    custom_metadata_fn = metadata_module.get_custom_metadata

                configs_valid.append(
                    LocalDatasetConfig(
                        id=pre_encoded_dir_config_valid["id"],
                        path=pre_encoded_dir_path,
                        n_repeats=0,
                        custom_metadata_fn=custom_metadata_fn
                    )
                )

            val_set = PreEncodedDataset(
                configs_valid,
                latent_crop_length=latent_crop_length,
                min_length_sec=min_length_sec,
                max_length_sec=max_length_sec,
                random_crop=random_crop,
                latent_extension=latent_extension
            )

            val_dl = torch.utils.data.DataLoader(
                val_set,
                batch_size,
                shuffle=False,
                num_workers=num_workers,
                persistent_workers=True,
                pin_memory=True,
                drop_last=dataset_config.get("drop_last", False),
                collate_fn=collation_fn,
            )
            list_valid_dl.append(val_dl)

        return train_dl, list_valid_dl

    elif dataset_type in ["wds", "s3"]:
        train_dataset_entries = dataset_config.get("datasets", None)
        assert train_dataset_entries is not None, "WebDataset configuration must be specified in datasets"

        wds_configs = create_webdataset_configs(
            train_dataset_entries,
            dataset_type,
            s3_streaming_defaults=dataset_config.get("s3_streaming", None),
        )

        train_dl = WebDatasetDataLoader(
            wds_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000),
            pre_encoded=dataset_config.get("pre_encoded", False),
            latent_crop_length=dataset_config.get("latent_crop_length", None),
            latent_extension=dataset_config.get("latent_extension", "npy"),
            min_length_sec=dataset_config.get("min_length_sec", None),
            max_length_sec=dataset_config.get("max_length_sec", None),
            resampled_shards=dataset_config.get("resampled_shards", True),
        ).data_loader

        datasets_valid = dataset_config.get("datasets_valid", None)
        list_valid_dl = []

        if datasets_valid is not None:
            if len(datasets_valid) > 0 and isinstance(datasets_valid[0], dict):
                datasets_valid = [datasets_valid]

            for dataset_valid_group in datasets_valid:
                valid_configs = create_webdataset_configs(
                    dataset_valid_group,
                    dataset_type,
                    s3_streaming_defaults=dataset_config.get("s3_streaming", None),
                )
                valid_dl = WebDatasetDataLoader(
                    valid_configs,
                    sample_rate=sample_rate,
                    sample_size=sample_size,
                    batch_size=batch_size,
                    random_crop=dataset_config.get("random_crop_valid", False),
                    num_workers=num_workers,
                    persistent_workers=True,
                    pin_memory=True,
                    force_channels=force_channels,
                    epoch_steps=dataset_config.get("epoch_steps_valid", dataset_config.get("epoch_steps", 2000)),
                    pre_encoded=dataset_config.get("pre_encoded", False),
                    latent_crop_length=dataset_config.get("latent_crop_length", None),
                    latent_extension=dataset_config.get("latent_extension", "npy"),
                    min_length_sec=dataset_config.get("min_length_sec", None),
                    max_length_sec=dataset_config.get("max_length_sec", None),
                    resampled_shards=dataset_config.get("resampled_shards_valid", False),
                    drop_last=False,
                ).data_loader
                list_valid_dl.append(valid_dl)

        return train_dl, list_valid_dl

    else:
        raise NotImplementedError
