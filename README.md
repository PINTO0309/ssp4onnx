# ssp4onnx
**S**imple **Sp**lit for ONNX. A simple tool that automatically splits ONNX models of specified weight sizes.

[![Downloads](https://static.pepy.tech/personalized-badge/ssp4onnx?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/ssp4onnx) ![GitHub](https://img.shields.io/github/license/PINTO0309/ssp4onnx?color=2BAF2B) [![PyPI](https://img.shields.io/pypi/v/ssp4onnx?color=2BAF2B)](https://pypi.org/project/ssp4onnx/)  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/ssp4onnx)

## Usage
```bash
pip install ssp4onnx

ssp4onnx -i model.onnx --auto_split_max_size 1GB
```

Options:
- `-i/--input_onnx_file`: input ONNX file (required)
- `-o/--output_dir`: output directory (optional, default: same directory as input)
- `-s/--auto_split_max_size`: target partition size, supports `KB`, `MB`, `GB` (default: `100MB`)

  <img width="539" height="330" alt="image" src="https://github.com/user-attachments/assets/6779d186-3f4c-4f5c-b860-49ffc2843967" />

## Sample reuslts

  |Split.1 (1.1GB)|Split.2 (1.1GB)|Split.3 (0.4GB)|
  |:-:|:-:|:-:|
  |<img width="615" height="656" alt="20260131213930" src="https://github.com/user-attachments/assets/6e622d15-e1d6-422f-aee5-d8a21c1c10fc" />|<img width="651" height="669" alt="20260131213947" src="https://github.com/user-attachments/assets/1a300e8e-3bd2-4994-b1e6-713837b2ea51" />|<img width="640" height="641" alt="20260131214003" src="https://github.com/user-attachments/assets/d12add6b-a390-41ec-91bc-69a928910521" />|

