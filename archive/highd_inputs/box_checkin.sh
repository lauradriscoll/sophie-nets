#!/bin/sh  
while true  
do  
rclone copy /home/laura/code/multitask-nets/stepnet/data/crystals/softplus/ /home/laura/data/rnn/multitask/crystals/softplus/
rclone copy /home/laura/data/rnn/multitask/crystals stanford_box:multitask/crystals --transfers 10 --tpslimit 10 -P
sleep 300
done
