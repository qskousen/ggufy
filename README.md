# ggufy
A lightweight and efficient tool to convert tensor formats.

ggufy aims to be fast and memory efficient. For reference, on my 9800X3D, it can convert sd 1.5 from safetensors (4.9 GB checkpoint) to q3_k gguf (skipping sensitivity) in about 6 seconds, using a max of about 252 MB of memory.

ggufy currently supports safetensors and gguf files.

ggufy is currently very targeted towards image diffusion models, specifically converting from safetensors to various gguf quantizations.

ggufy is a work in progress, in the early stages. It can convert the most common types of image models from safetensors ("checkpoint" style or unet style), but "sensitivity" files only exist for sd 1.5 so far.

### Todos:

- [ ] allow opening st or gguf non-existing path and use for writing (seperate instances for read and write) (gguf done)
- [x] generate "quantization sensitivity" file, weight tensors 1-100 on how much quantization affects them (done for sd1.5, more to come)
- [x] allow to set output directory and output file when converting
- [x] starting with q8_0, support actual quantization
- [ ] allow converting model, encoders, vae by option
- [ ] allow setting alignment to something other than 32

I initially intended to have this all in pure zig, but now it includes ggml c/c++ code for quantization. I did actually get a working q8_0 implementation in zig (you can find it if you look back through the commits) but got stuck on figuring out q5_0 and decided to just pull in ggml and use that.

## Usage

### Basic Commands

ggufy supports several commands for inspecting and converting model files:

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


#### Generate Template

Export a GGUF model's structure as a JSON template (useful for conversion verification or copying quantization levels and metadata from a known working file):

```bash
ggufy template model.gguf
```


This creates a `template.json` file containing the tensor names, shapes, and types, as well as metadata.

### Converting Models

The `convert` command is the main feature of ggufy, allowing you to convert safetensors models to GGUF format with various quantization levels.

#### Basic Conversion

Convert a safetensors model to GGUF with default settings:

```bash
ggufy convert model.safetensors
```

This will:
- Auto-detect the model architecture (SD1.5, SDXL, Flux, etc.)
- Leave the tensor data types unchanged where possible
- Save to `model-F16.gguf` in the same directory

#### Specify Quantization Level

Convert to a specific quantization type:

```bash
# Convert to Q4_K (good balance of quality/size)
ggufy convert --datatype q4_k model.safetensors
# Convert to Q8_0 (near-lossless)
ggufy convert --datatype q8_0 model.safetensors
# Convert to Q2_K (maximum compression)
ggufy convert --datatype q2_k model.safetensors
```

Available output types:
- `f32` - 32-bit float (uncompressed)
- `bf16` - 16-bit float (brainfloat 16, less precision loss than f16)
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

#### Template-based Conversion

Use a pre-defined template to ensure specific tensor shapes and types:

```bash
# First, generate a template from a working GGUF
ggufy template reference-model.gguf
# Then use it to convert another model
ggufy convert --template template.json model.safetensors
```

This is useful when you need to match a specific model format exactly.

### Sensitivity-Aware Quantization

For supported architectures (currently SD1.5), ggufy can use sensitivity data to apply different quantization levels to different layers (for supported models, sensitivity is enabled by default:

```bash
ggufy convert --datatype q4_k model.safetensors
```

When sensitivity data is available:
- **Low sensitivity layers** (5-30): Quantized to the target level (e.g., Q4_K)
- **Medium sensitivity layers** (30-70): Quantized to a higher precision (e.g., Q6_K or Q8_0)
- **High sensitivity layers** (70-95): Kept at Q8_0 or source precision
- **Critical layers** (95+): Always kept at source precision (F16/F32)

The `--aggressiveness` option (default: 50) controls how aggressively layers are quantized:
- Higher values (50-100): More aggressive, most layers stay near target quantization (whatever quantization you passed in -- e.g. Q4_K)
- Higher values (1-50): More conservative, sensitive layers quickly upgraded to higher precision

Sensitivity quantization can be turned off with `--skip-sensitivity` or `-s`.

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

### Options Reference

```
-h, --help              Display help and exit
-d, --datatype          Target quantization type (default: source datatype) 
-f, --filetype          Target file format (default: gguf) Options: gguf, safetensors 
-t, --template          Use a JSON template for conversion 
-o, --output-dir        Output directory (default: same as source) 
-n, --output-name       Output filename without extension (default: source name + datatype) 
-j, --threads           Number of threads for quantization (default: CPU cores - 2)
-a, --aggressiveness    Aggressiveness of sensitivity-aware quantization (default: 50)
-s, --skip-sensitivity  Skip sensitivity-aware quantization, quantizing all available layers to target datatype

Commands: 
header - Display file header information 
tree - Display tensor hierarchy 
metadata - Display all metadata key-value pairs 
convert - Convert model format/quantization 
template - Export GGUF structure to JSON template
```

## Acknowledgements

- [ggml](https://github.com/ggml-org/ggml) ggufy uses ggml for quantization
- [ComfyUI-GGUF by city96](https://github.com/city96/ComfyUI-GGUF) for helping me to understand a lot about how the quantization works, as well as architecture detection
- [llama.cpp.zig](https://github.com/Deins/llama.cpp.zig) for helping get ggml compiling in zig
- [gguf-docs](https://github.com/iuliaturc/gguf-docs) for helping me understand quantization