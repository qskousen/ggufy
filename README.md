# ggufy
A lightweight and efficient tool to convert tensor formats.

ggufy currently supports safetensors and gguf files.

todos:

- [ ] allow opening st or gguf non-existing path and use for writing (seperate instances for read and write) (gguf done)
- [ ] generate "importance matrix" file from a template, weight tensors 100-0 on how much quantization affects them
- [x] allow to set output directory and output file when converting
- [ ] starting with q8_0, support actual quantization
- [ ] allow converting model, encoders, vae by option
- [ ] allow setting alignment to something other than 32