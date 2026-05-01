#!/usr/bin/env bash

set -e

shopt -s nullglob
trap 'shopt -u nullglob' EXIT

if [[ $# -eq 0 ]] ; then
    echo 'Please give two arguments to the script: [input_filename] [output_filename].'
    exit 0
fi

# On Mac OS - brew install coreutils
# On linux: split

TMP_DIR=/tmp/openie/large_corpus

rm -rf $TMP_DIR
mkdir -p $TMP_DIR

if [[ ! -s "$1" ]]; then
    echo "Input file $1 is empty; creating an empty output at $2."
    mkdir -p "$(dirname "$2")"
    : > "$2"
    exit 0
fi

if [ "$(uname)" == "Darwin" ]; then
    gsplit -b 10k --numeric-suffixes "$1" "${TMP_DIR}/small_"
else
    split -b 10k --numeric-suffixes "$1" "${TMP_DIR}/small_"
fi

small_files=(${TMP_DIR}/small_*)
num_files=${#small_files[@]}
var=1
for file in "${small_files[@]}"
do
    if [[ -f $file ]]; then
        echo "(${var} / ${num_files}) python main.py -f $file > $file.out"
        python main.py -f "$file" > "$file.out"
        var=$((var + 1))
    fi
done

out_files=(${TMP_DIR}/*.out)
if (( ${#out_files[@]} == 0 )); then
    : > "$2"
else
    cat "${out_files[@]}" > "$2"
fi
echo "Redirected the output to $2"
