#!/bin/bash
bash run_$1.sh &
while true
do
  sleep 100
  # RUNNING=$(lsof /dev/nvidia2 | grep python  | awk '{print $2}' | sort -u | wc -l)
#   RUNNING=$(ps aux | grep tools/traintest.py | grep -v grep | awk '{print $2}' | wc -l)
  b=$(nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}')
  for i in $(echo $b | tr ";" '\n'); do
    # echo $i
    if (($i < 200)); then
      pkill python
      sleep 10
      pkill python
      bash run_$1.sh &
      sleep 100
      # process "$i"
    fi
  done
  # if (($RUNNING < $2)); then
  #   pkill python
  #   for i in $(ps aux | grep tools/traintest.py | grep -v grep | awk '{print $2}'); do kill -9 $i; done
  #   bash run_$1.sh &
  # fi
done

# a=4
# while true; do
#   b=$(nvidia-smi --query-gpu=memory.used --format=csv|grep -v memory|awk '{print $1}')
#   # echo $b
#   for i in $(echo $b | tr ";" '\n'); do
#     # echo $i
#     if (($i < 200)); then
#       echo $i
#       # process "$i"
#     fi
#   done
#   # [ $b -gt $a ] && a=$b && echo $a
#   sleep .5
# done

# wait 100;
# while true
# do
#   echo 1
#   sleep 1
# done
# v
# for i in $(sudo lsof /dev/nvidia2 | grep python  | awk '{print $2}' | sort -u)

# for i in $(sudo lsof /dev/nvidia2 | grep python  | awk '{print $2}' | sort -u); do kill -9 $i; done

# RUNNING=$(ps aux | grep tools/traintest.py | grep -v grep | awk '{print $2}' | wc -l)
# #$(sudo lsof /dev/nvidia2 | grep python  | awk '{print $2}' | sort -u | wc -l)
