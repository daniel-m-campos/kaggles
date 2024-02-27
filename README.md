# Kaggles
A collection of Kaggle competitions I have participated in.

# Usage
## 1. Create Virtual Environment
```bash
script/create_venv.sh
```

## 2. Create Competition Directory
```bash
cd kaggles
mkdir -p <competition_name>/kernels
```

## 3. Download Competition Data
```bash
script/download_data.sh
```

## 4. Pull Competition Kernels
```bash
script/pull_kernels.sh <competition_name>
```

## 5. Push Competition Kernels
```bash
kaggle k push -p <competition_name>/kernels/<kernel_dir>
```
