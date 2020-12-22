#!/bin/bash
set -e

checkpoint=model-weights/rr/model-30501.pkl.gz
batchsize=16
outdir=examples

if [ ! -d $outdir ]; then
    mkdir -p $outdir
fi

while true; do
    batch=$(find animal-faces/|sort -R|head -$batchsize|awk -vORS=, '{ print $0 }' | sed 's/,$/\n/')
    checkpoint=$(find model-weights/rr/*.pkl.gz|sort -R|head -1)
    out=$(basename $(basename $checkpoint) .pkl.gz)
    out=$outdir/$out-example.jpg

    echo $out
    ./pixace.py \
    --mode=predict \
    --image_size=32 \
    --predict_input=$batch \
    --cut=512 \
    --temps=1 \
    --checkpoint=$checkpoint \
    --out=$out
done
