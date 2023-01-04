# Silent Killer

## Background
The project is the official implementation of Silent-Killer data poisoning paper.

## Requirements

## Run
python silent_killer.py --trigger_type additive --eps 0.062745
python silent_killer.py --trigger_type adaptive_patch --eps 1

To specify source label and target label use `-s` and `-t`. 
To log results to wandb specify the entity, for example `--entity jhon`
To use a pretrained model for crafting rather than retrain one from scratch in the beginning of the crafting use `--model path`
