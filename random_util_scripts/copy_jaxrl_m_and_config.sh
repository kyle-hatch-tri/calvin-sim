#!/bin/bash

cp -r /home/kylehatch/Desktop/hidql/bridge_data_v2/jaxrl_m/* /home/kylehatch/Desktop/hidql/calvin-sim/external/jaxrl_m/jaxrl_m
cp /home/kylehatch/Desktop/hidql/bridge_data_v2/experiments/configs/susie/calvin/configs/gcbc_train_config.py /home/kylehatch/Desktop/hidql/calvin-sim/calvin_models/calvin_agent/evaluation

echo "Don't forget to comment out the last line in /home/kylehatch/Desktop/hidql/calvin-sim/calvin_models/calvin_agent/evaluation/gcbc_train_config.py"