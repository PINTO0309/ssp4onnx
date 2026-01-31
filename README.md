# ssp4onnx
**S**imple **Sp**lit for ONNX. A simple tool that automatically splits ONNX models of specified weight sizes.

## Usage
```bash
ssp4onnx -i model.onnx --auto_split_max_size 100MB
```

Options:
- `-i/--input_onnx_file`: input ONNX file (required)
- `-o/--output_dir`: output directory (optional, default: same directory as input)
- `-s/--auto_split_max_size`: target partition size, supports `KB`, `MB`, `GB` (default: `100MB`)

<img width="539" height="330" alt="image" src="https://github.com/user-attachments/assets/6779d186-3f4c-4f5c-b860-49ffc2843967" />
