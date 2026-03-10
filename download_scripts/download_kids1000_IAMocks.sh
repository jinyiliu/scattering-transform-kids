#!/usr/bin/env bash

BASE_URL="http://cuillin.roe.ac.uk/~bengib/MassMaps4JHD/KiDS1000"

curl -s "$BASE_URL/" | grep -oP 'href="\K[^"]+' | grep IA | grep / | sed 's|/$||' > IA_directories.txt

while read -r dir; do
    wget -r -np -nH --cut-dirs=3 -R "index.html*" \
    "$BASE_URL/$dir/"
done < IA_directories.txt

rm IA_directories.txt