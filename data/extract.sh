#!/bin/bash

RAW_FOLDER=./raw/
mkdir -p RAW_FOLDER

for f in `ls archives/*.tar.gz`; do
    echo -n "Extracting $f..."
    tar -zxf $f -C $RAW_FOLDER
    echo "... done!" 
done

for f in `ls archives/*.zip`; do
    echo -n "Extracting $f..."
    unzip -q -d $RAW_FOLDER $f
    echo "... done!"
done
