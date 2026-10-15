"""Write HunyuanVideo tensor-name fixtures for arch detection, from the headers
of the two published layouts. Only the headers are fetched, with range requests.

  src/test_fixtures/hyvid.json        city96's GGUF: Tencent's original names
  src/test_fixtures/hyvid.comfy.json  Comfy-Org's safetensors: ComfyUI's names, "model.model." prefix
"""
import json
import struct
import urllib.request

GGUF_URL = "https://huggingface.co/city96/HunyuanVideo-gguf/resolve/main/hunyuan-video-t2v-720p-Q4_0.gguf"
ST_URL = "https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/resolve/main/split_files/diffusion_models/hunyuan_video_t2v_720p_bf16.safetensors"


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req).read()


def safetensors_names(url):
    n = struct.unpack("<Q", fetch(url, 0, 7))[0]
    header = json.loads(fetch(url, 8, 7 + n))
    header.pop("__metadata__", None)
    return sorted(header)


def gguf_names(url):
    # The tensor table sits right after the KV section, well inside the first 8 MiB.
    b = fetch(url, 0, (8 << 20) - 1)
    pos = 4
    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def read(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, b, pos)[0]
        pos += struct.calcsize("<" + fmt)
        return v

    def string():
        nonlocal pos
        n = read("Q")
        s = b[pos:pos + n].decode()
        pos += n
        return s

    def value(t):
        if t == 8:
            return string()
        if t == 9:
            et, n = read("I"), read("Q")
            return [value(et) for _ in range(n)]
        return read(scalar[t])

    read("I")
    n_tensors, n_kv = read("Q"), read("Q")
    for _ in range(n_kv):
        string()
        value(read("I"))
    names = []
    for _ in range(n_tensors):
        names.append(string())
        n_dims = read("I")
        for _ in range(n_dims):
            read("Q")
        read("I")
        read("Q")
    return sorted(names)


for path, names in (
    ("src/test_fixtures/hyvid.json", gguf_names(GGUF_URL)),
    ("src/test_fixtures/hyvid.comfy.json", safetensors_names(ST_URL)),
):
    with open(path, "w") as f:
        json.dump(names, f, indent=1)
        f.write("\n")
    print(f"{path}: {len(names)} tensors")
