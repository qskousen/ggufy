# ggufy
A lightweight and efficient tool to convert tensor formats.

ggufy:
- is a single-file executable written in zig, for linux, windows, and macos (arm64 and x86_64)
- is fast and memory efficient
- supports converting from safetensors to various gguf quantizations
- supports converting with "[quantization sensitivity](https://github.com/qskousen/ggufy#sensitivity-aware-quantization)" files (some architectures built-in)
- currently targets image diffusion models (SD1.5, SDXL, etc.)

Download pre-built executables [on the releases page](https://github.com/qskousen/ggufy/releases)

### Supported architectures

This table lists the architectures that ggufy can convert, and whether they have sensitivity data available.

| Architecture       | Supported | Sensitivity Data |
|--------------------|-----------|------------------|
| SD1.5              | ✅         | ✅                |
| SDXL               | ✅         | ✅                |
| Flux               | ✅         | ❌                |
| Lumina2 (ZiT, ZiB) | ✅         | ❌                |
| Aura               | ✅         | ❌                |
| HiDream            | ✅         | ❌                |
| Cosmos             | ✅         | ❌                |
| LTXV               | ✅         | ❌                |
| Hyvid              | ✅         | ❌                |
| WAN                | ✅         | ❌                |
| SD3                | ✅         | ❌                |
| Qwen               | ✅         | ❌                |

### Todos:

- [ ] allow opening st or gguf non-existing path and use for writing (seperate instances for read and write) (gguf done)
- [x] generate "quantization sensitivity" file, weight tensors 1-100 on how much quantization affects them
- [x] allow to set output directory and output file when converting
- [x] starting with q8_0, support actual quantization
- [ ] allow converting model, encoders, vae by option
- [ ] allow setting alignment to something other than 32

I initially intended to have this all in pure zig, but now it includes ggml c/c++ code for quantization. I did actually get a working q8_0 implementation in zig (you can find it if you look back through the commits) but got stuck on figuring out q5_0 and decided to just pull in ggml and use that.

## Installation

ggufy is a single-file executable, available for download from the [releases page](https://github.com/qskousen/ggufy/releases).
Download the version appropriate for your system, extract it, and place it somewhere in your PATH; alternatively, run it directly from wherever you extracted it.

## Usage

The primary use case for ggufy is converting models from safetensors to gguf format.

There are three main ways to do that:
- Using the `convert` command by itself,
- Using the `template` command to generate a JSON template for a model and using it with `convert`,
- Using the `convert` command with a sensitivity file to perform sensitivity-aware quantization.

### `Convert` command

```bash
ggufy convert [OPTIONS] <input-file>
# Example:
ggufy convert model.safetensors -d q4_k
```
The `convert` command takes a single required argument, the input file to convert.
It can also take several optional arguments, such as `-d` for specifying the quantization type, `-n` for specifying an output file, and `-o` for specifying the output directory.

```bash
ggufy convert model.safetensors -d q4_k -n my-model-q4-k -o ./converted/
```

### `Template` command

You can create a JSON template from an existing GGUF file that you want to copy, and use it with the `convert` command to ensure that the output file has the same structure and metadata as the template.
This can be useful when ggufy doesn't recognize a model architecture, or you want a specific quantization level for each layer.

```bash
ggufy template existing.gguf
# exports to `template.json`; now you can use it to convert another model:
ggufy convert model.safetensors -t template.json
```

### `convert` with sensitivity

Sensitivity-aware quantization is enabled by default for supported architectures.
You can disable it with the `--skip-sensitivity` or `-x` flag.
You can also specify an aggressiveness level with the `--aggressiveness` or `-a` flag, from 0-100.
You can also pass in a custom sensitivity file with the `--sensitivities` or `-s` flag.

Read more about sensitivity-aware quantization [below](#sensitivity-aware-quantization).

```bash
# less aggressive; more precision, larger filesize
ggufy convert sdxl.safetensors -d q4_k -a 25
# more aggressive; less precision, smaller filesize
ggufy convert sdxl.safetensors -d q4_k -a 75
# custom sensitivity file
ggufy convert sdxl.safetensors -d q4_k -s custom-sensitivity.json
```

## Other Commands

ggufy supports several other commands for inspecting model files:

#### View File Header

Display basic information about a model file:

```bash
# View safetensors header
ggufy header model.safetensors
# View GGUF header
ggufy header model.gguf
```

#### View Tensor Tree

Display the hierarchical structure of tensors in a model (only implemented for safetensors for now):

```bash
ggufy tree model.safetensors
```

This shows tensors organized by their layer structure, making it easy to understand the model architecture.

#### View Metadata

Display all metadata key-value pairs stored in the model:

```bash
# View safetensors metadata
ggufy metadata model.safetensors
# View GGUF metadata
ggufy metadata model.gguf
```

### Quantization Level

ggufy supports converting models to a variety of quantization levels via the `--datatype` (`-d`) flag.

```bash
# Convert to Q4_K (good balance of quality/size)
ggufy convert --datatype q4_k model.safetensors
# Convert to Q8_0 (near-lossless)
ggufy convert -d q8_0 model.safetensors
# Convert to Q2_K (maximum compression, highest quality loss)
ggufy convert -d q2_k model.safetensors
```

Available output types:
- `f32` - 32-bit float (uncompressed)
- `bf16` - 16-bit float (brainfloat 16, higher precision than f16)
- `f16` - 16-bit float (half precision)
- `q8_0` - 8-bit quantization (equivalent to f16 quality)
- `q5_0` - 5-bit quantization
- `q5_1` - 5-bit quantization
- `q4_0` - 4-bit quantization
- `q4_1` - 4-bit quantization
- `q6_k` - 6-bit quantization
- `q5_k` - 5-bit quantization
- `q4_k` - 4-bit quantization
- `q3_k` - 3-bit quantization
- `q2_k` - 2-bit quantization

The `_k` types have a higher compression ratio than their non-k counterparts, at a small cost of quality.

Importance matrix quantization (`iq2_xxs`, `iq4_xs`, etc) is not yet supported. `f64` is not supported and is automatically downcast to `f32` when converting because ComfyUI, at least, does not support f64 values in gguf files at this time.

#### Custom Output Location

Specify output directory and filename:

```bash
# Set output directory
ggufy convert --datatype q4_k --output-dir ./converted model.safetensors
# Set custom output name (without extension)
ggufy convert --datatype q4_k --output-name my-model-v2 model.safetensors
# Combine both
ggufy convert --datatype q4_k --output-dir ./models --output-name flux-dev-q4 flux-dev.safetensors
```

#### Multi-threaded Conversion

Control the number of threads used for quantization (default: CPU cores - 2):

```bash
# Use 8 threads
ggufy convert --datatype q4_k --threads 8 model.safetensors
# Use single thread
ggufy convert --datatype q4_k --threads 1 model.safetensors
```

### Sensitivity-Aware Quantization

For supported architectures, ggufy can use sensitivity data to apply different quantization levels to different layers.
This can significantly improve the quality of the model at the cost of slightly larger file sizes.
For supported models, sensitivity is enabled by default.

Generating sensitivity data requires generating a large number of images, which can take a long time.
More architectures will be added as the sensitivity data is generated for them.

```bash
ggufy convert --datatype q4_k sd1.5.safetensors
```

When sensitivity data is available:
- **Low sensitivity layers** (5-30): Quantized to the target level (e.g., Q4_K)
- **Medium sensitivity layers** (30-70): Quantized to a higher precision (e.g., Q6_K or Q8_0)
- **High sensitivity layers** (70-95): Kept at Q8_0 or source precision
- **Critical layers** (95+): Always kept at source precision (F16/F32)

The `--aggressiveness` option (default: 50) controls how aggressively layers are quantized:
- Higher values (50-100): More aggressive, most layers stay near target quantization (whatever quantization you passed in -- e.g. Q4_K)
- Higher values (1-50): More conservative, sensitive layers quickly upgraded to higher precision

```bash
ggufy convert --datatype q4_k sd1.5.safetensors --aggressiveness 25
```

Sensitivity quantization can be turned off with `--skip-sensitivity` or `-x` if you want to use the default quantization levels for all layers.

```bash
ggufy convert --datatype q4_k sd1.5.safetensors --skip-sensitivity
```

You can also specify a custom sensitivity file with `--sensitivities` or `-s`. For an example sensitivity file, look in the `src/sensitivities` directory.

```bash
ggufy convert --datatype q4_k sd1.5.safetensors --sensitivities custom-sensitivity.json
```

By default, the quantization levels will be chosen based on the target datatype: `QX_K` variants for a `_k` target datatype, `QX_1` variants for a `_1` target datatype, and `QX_0` variants for a `_0` target datatype.
Non-quantized datatypes (up to the source datatype) and Q8_0 are allowed in all cases.
You can override this by specifying which quantization types to enable with `--use-quant-types` or `-q` followed by a list seperated by commas. The choices are `0`, `1`, `k` to select those quantization types.

For example, if you want to use K types and 0 types, you can do:

```bash
ggufy convert --datatype q4_k --use-quant-types k,0 sd1.5.safetensors
```

### Complete Examples

```bash
# Inspect a model before converting
ggufy header sd15-model.safetensors ggufy tree sd15-model.safetensors
# Convert SD1.5 to Q4_K with sensitivity-aware quantization
ggufy convert --datatype q4_k --output-dir ./output sd15-model.safetensors
# Convert SDXL to Q8_K (near-lossless) with custom threading
ggufy convert --datatype q8_k --threads 12 --output-name sdxl-q8 sdxl-model.safetensors
# Convert Flux model to Q5_K with custom output location
ggufy convert --datatype q5_k --output-dir ./converted-models --output-name flux-dev-q5 --threads 16 flux-dev.safetensors
# Generate template from GGUF for verification
ggufy template converted-model.gguf
# Convert using template for exact shape matching
ggufy convert --template template.json --datatype q4_k source-model.safetensors
```

### Options and Commands Reference

```
-h, --help              Display help and exit
-d, --datatype          Target quantization type (default: source datatype) 
-f, --filetype          Target file format (default: gguf) Options: gguf, safetensors 
-t, --template          Use a JSON template for conversion 
-o, --output-dir        Output directory (default: same as source) 
-n, --output-name       Output filename without extension (default: source name + datatype) 
-j, --threads           Number of threads for quantization (default: CPU cores - 2)
-a, --aggressiveness    Aggressiveness of sensitivity-aware quantization (default: 50)
-x, --skip-sensitivity  Skip sensitivity-aware quantization, quantizing all available layers to target datatype
-s, --sensitivities     Path to custom sensitivity file 
-q, --use-quant-types   Comma-separated list of quantization types to use (0,1,k) (default: based on target datatype)

Commands: 
header - Display file header information 
tree - Display tensor hierarchy 
metadata - Display all metadata key-value pairs 
convert - Convert model format/quantization 
template - Export GGUF structure to JSON template
```

## Building

ggufy is built with zig 0.15.2. Clone the repository, with submodules:

```bash
git clone --recurse-submodules https://github.com/qskousen/ggufy.git
```

Build with zig:
```bash
zig build
```

## Acknowledgements

- [ggml](https://github.com/ggml-org/ggml) ggufy uses ggml for quantization
- [ComfyUI-GGUF by city96](https://github.com/city96/ComfyUI-GGUF) for helping me to understand a lot about how the quantization works, as well as architecture detection
- [llama.cpp.zig](https://github.com/Deins/llama.cpp.zig) for helping get ggml compiling in zig
- [gguf-docs](https://github.com/iuliaturc/gguf-docs) for helping me understand quantization
- Random German who helped to get ggufy working on MacOS for arm64 and x86_64