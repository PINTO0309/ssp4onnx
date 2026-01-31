# ssp4onnx
**S**imple **Sp**lit for ONNX. A simple tool that automatically splits ONNX models of specified weight sizes.

[![Downloads](https://static.pepy.tech/personalized-badge/ssp4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/ssp4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/ssp4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/ssp4onnx?color=2BAF2B)](https://pypi.org/project/ssp4onnx/)  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/ssp4onnx)

## Usage
```bash
ssp4onnx -i model.onnx --auto_split_max_size 100MB
```

Options:
- `-i/--input_onnx_file`: input ONNX file (required)
- `-o/--output_dir`: output directory (optional, default: same directory as input)
- `-s/--auto_split_max_size`: target partition size, supports `KB`, `MB`, `GB` (default: `100MB`)

<img width="539" height="330" alt="image" src="https://github.com/user-attachments/assets/6779d186-3f4c-4f5c-b860-49ffc2843967" />
