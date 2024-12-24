#!/bin/bash

_SRC=$(realpath "${1}")
_DST=$(realpath "${2}")

echo "Source dir: ${_SRC}"
echo "Destination dir: ${_DST}"

mkdir -pv "${_DST}"/{train,test,mask,roi}/{good,bad}

echo "Copying files: ${_SRC}/train/good -> ${_DST}/train"
rsync -hhhHpPrtacI --quiet "${_SRC}/train/good" "${_DST}/train" && echo DONE || exit 1

echo "Copying files: ${_SRC}/test/good -> ${_DST}/test"
rsync -hhhHpPrtacI --quiet "${_SRC}/test/good" "${_DST}/test" && echo DONE || exit 1

# Iterate over directories in "${_SRC}/test"
for dir in $(find "${_SRC}/test" -mindepth 1 -maxdepth 1 -type d); do
    # Skip if not a directory or if the directory name is "good"
    [ -d "$dir" ] || continue
    [ "$(basename "$dir")" == "good" ] && continue

    # Print files in subdirectories
    for f in $(find "${_SRC}/test"/$(basename "$dir") -type f); do
        # Get the file name without extension
        file_name=$(basename "${f}" | sed 's/\.[^.]*$//')

        # Get the file extension without '.'
        file_extension=$(basename "${f}" | grep -oE '\.[^.]+$' | sed 's/^\.//')

        # Get the path without the file name
        file_path=$(dirname "${f}")

        # Get the last directory in the path
        last_directory=$(basename "$file_path")

        new_name=${last_directory}${file_name}"."${file_extension}

        echo "Copying file: ${f} -> ${_DST}/test/bad/${new_name}"
        rsync -hhhHpPrtacI --quiet "${f}" "${_DST}/test/bad/${new_name}"
    done
done


# Iterate over directories in "${_SRC}/test"
for dir in $(find "${_SRC}/ground_truth" -mindepth 1 -maxdepth 1 -type d); do
    # Skip if not a directory or if the directory name is "good"
    [ -d "$dir" ] || continue
    [ "$(basename "$dir")" == "good" ] && continue

    # Print files in subdirectories
    for f in $(find "${_SRC}/ground_truth"/$(basename "$dir") -type f); do
        # Get the file name without extension
        file_name=$(basename "${f}" | sed 's/\.[^.]*$//')

        # Get the file extension without '.'
        file_extension=$(basename "${f}" | grep -oE '\.[^.]+$' | sed 's/^\.//')

        # Get the path without the file name
        file_path=$(dirname "${f}")

        # Get the last directory in the path
        last_directory=$(basename "$file_path")

        new_name=${last_directory}${file_name}"."${file_extension}

        echo "Copying file: ${f} -> ${_DST}/mask/bad/${new_name}"
        rsync -hhhHpPrtacI --quiet "${f}" "${_DST}/mask/bad/${new_name}"
    done
done

