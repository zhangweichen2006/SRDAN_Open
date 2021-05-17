######### resume code after program crash (with clear leftover GPU memory) ####

#!/bin/bash
bash run_$1.sh &
while true
do
  sleep 100
  b=$(nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}')
  for i in $(echo $b | tr ";" '\n'); do
    # echo $i
    if (($i < 200)); then # when one gpu crash and the other two
      pkill python
      sleep 10
      pkill python
      bash run_$1.sh &
      sleep 100
    fi
  done
done