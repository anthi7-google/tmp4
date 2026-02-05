#!/bin/bash

mn_list=("FCRMixer")

#"etth1" "ettm1" "etth2" "ettm2" "illness" "weather" "ele"
names=("etth1" "ettm1" "etth2" "ettm2" "illness" "weather" "ele")

gpu=0

for mn in "${mn_list[@]}"
do
for name in "${names[@]}"
do
  bash "scripts/_FCR_${name}.sh" --mn "$mn" --gpu "$gpu"
done
done