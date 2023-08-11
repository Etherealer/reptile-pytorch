#!/bin/bash
#
# Fetch Mini-ImageNet.
#

IMAGENET_URL=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

set -e

mkdir tmp
trap 'rm -r tmp' EXIT

if [ ! -d data ]; then
   mkdir data
fi

if [ ! -d data/miniimagenet ]; then
   mkdir tmp/miniimagenet
   for subset in train test val; do
       mkdir "tmp/miniimagenet/$subset"
       echo "Fetching Mini-ImageNet $subset set ..."
       for csv in metadata/miniimagenet/"$subset"/*.csv; do
           echo "Fetching : $(basename "${csv%.csv}")"
           dst_dir="tmp/miniimagenet/$subset/$(basename "${csv%.csv}")"
           mkdir "$dst_dir"
           while read -r entry; do
               name=$(echo "$entry" | cut -f 1 -d ,)
               range=$(echo "$entry" | cut -f 2 -d ,)
               curl -s -H "range: bytes=$range" $IMAGENET_URL > "$dst_dir/$name" &
           done < "$csv"
           wait
       done
   done
   mv tmp/miniimagenet data/miniimagenet
fi
