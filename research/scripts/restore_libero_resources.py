"""Resume LIBERO resources on /data, using HF mirror and ModelScope."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def main():
    """Download independent resources concurrently, preserving resumable partials."""
    # Domestic endpoints work directly on this host; inherited SOCKS settings fail.
    for key in ["ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]:
        os.environ.pop(key, None)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HOME"] = "/data/zhoukai/pretrain/hf_cache"
    os.environ["MODELSCOPE_CACHE"] = "/data/zhoukai/pretrain/modelscope_cache"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download as hf_download
    from modelscope.hub.snapshot_download import snapshot_download as ms_download

    jobs = {
        "dataset": lambda: hf_download(
            "lerobot/libero",
            repo_type="dataset",
            revision="a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4",
            local_dir="/data/zhoukai/dataset/libero",
            max_workers=4,
        ),
        "assets": lambda: hf_download(
            "lerobot/libero-assets",
            repo_type="dataset",
            revision="0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
            local_dir="/data/zhoukai/dataset/libero-assets",
            max_workers=4,
        ),
        "official": lambda: hf_download(
            "lerobot/pi05_libero_finetuned",
            revision="8e174154ef5f6c60a8da12ae99c303d8963138c1",
            local_dir="/data/zhoukai/pretrain/pi05_libero_finetuned",
            max_workers=4,
        ),
        "dino": lambda: ms_download(
            "facebook/dinov3-vits16-pretrain-lvd1689m",
            local_dir="/data/zhoukai/pretrain/dinov3-vits16-pretrain-lvd1689m",
            endpoint="https://www.modelscope.cn",
        ),
        "tokenizer": lambda: ms_download(
            "AI-ModelScope/paligemma-3b-pt-224",
            local_dir="/data/zhoukai/pretrain/paligemma-tokenizer",
            allow_file_pattern=["config.json", "tokenizer*", "special_tokens_map.json", "added_tokens.json"],
            endpoint="https://www.modelscope.cn",
        ),
    }
    errors = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(job): name for name, job in jobs.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                print(f"COMPLETE {name}: {future.result()}", flush=True)
            except Exception as error:
                errors.append(name)
                print(f"FAILED {name}: {error}", flush=True)
    if errors:
        raise RuntimeError(f"Rerun to resume failed downloads: {errors}")
    print("All resources restored.", flush=True)


if __name__ == "__main__":
    main()
