#!/bin/bash

case ${1} in 
    1) # run tf version
          python run_inference.py \
              --inference_request="$(cat configs/inference_training_sample2.pbtxt)" \
              --bounding_box 'start { x:0 y:0 z:0 } size { x:250 y:250 z:250 }'
        ;;
esac
