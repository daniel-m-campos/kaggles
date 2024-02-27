#!/bin/bash -e

if [ -z "$1" ]; then
    echo "Usage: $0 <competition_name>"
    exit 1
fi

COMPETITION_NAME=$1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SOURCE_ROOT=$DIR/..

if [ ! -d "$SOURCE_ROOT/$COMPETITION_NAME" ]; then
    mkdir -p "$SOURCE_ROOT/$COMPETITION_NAME/notebooks"
fi

kernel_list=$(kaggle kernels list --mine --competition $COMPETITION_NAME |
    awk 'NR>2 {print $1}' |
    grep "$USER")

for kernel in $kernel_list; do
    kernel_dir=$(echo $kernel | sed 's/.*\///')
    kaggle kernels pull \
        -p "$SOURCE_ROOT/$COMPETITION_NAME/kernels/$kernel_dir" \
        "$kernel" -m
done
