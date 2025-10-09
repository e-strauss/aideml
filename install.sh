#!/usr/bin/env bash
set -e

while read -r requirement; do
    [[ -z "$requirement" || "$requirement" == \#* ]] && continue

    echo "Installing: $requirement"
    pip install "$requirement" || break
done < requirements.txt
