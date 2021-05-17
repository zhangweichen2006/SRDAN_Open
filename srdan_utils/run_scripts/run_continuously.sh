#!/bin/bash
while true
do
  bash run_$1.sh
  pkill python
done