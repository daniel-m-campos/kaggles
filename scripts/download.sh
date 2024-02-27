#!/bin/bash -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SOURCE_ROOT=$DIR/..
cd $SOURCE_ROOT || exit

find . -type d -name kernels | while read -r d; do
    dir=$(dirname $d)
    dir_name=$(basename $dir)
    echo Found competition dir "$dir_name"
    cd $dir_name/kernels || exit
    if [ -d input ]; then
        echo Removing "$dir_name" input directory
        rm -rf input
    fi
    mkdir input
    kaggle competitions download -c "$dir_name" -p input
    unzip -o input/*.zip -d input
    rm -rf input/*{.zip,gz}
done
