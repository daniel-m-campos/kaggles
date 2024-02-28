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

# Utility Scripts
Kaggle supports [utility scripts](https://www.kaggle.com/discussions/product-feedback/91185) to help avoid code duplication but they can be a little tricky to set up with a repo. This is how it's done here:
1. Create a new kernel and add the utility code to it. E.g. `home-data-for-ml-course/src/`
1. Push the kernel to Kaggle. `kaggle k push -p home-data-for-ml-course/src/`
1. Go to Kaggle, open the utility kernel with the editor and click `File -> Set as utility script`.
1. For kernels that depend on the utility script, add the utility kernel to the `kernel_sources` list in the `kernel-metadata.json` file. E.g.
    ```json
    "kernel_sources": [
        "danielmcampos/house-price-utils"
    ],
    ```
    Alternatively,  click `File -> Add utility script` and select the utility script.
