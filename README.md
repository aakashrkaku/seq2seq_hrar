Official Implementation of "Sequence-to-Sequence Modeling for Action Identification at High Temporal Resolution"




### Dataset
The storke dataset will be available soon. For the public dataset such as breakfast and 50salads, please check the the official code from ASRF https://github.com/yiskw713/asrf

We follow the same setting to conduct data split.


### Experiments
Here is the example script
```
python3 main_seq2seq.py \
--val_fold 4 \
--dataset_name breakfast \
--model_name  breakfast_video \
--model_path   MODEL_SAVE_PATH  \
--path_root  DATA_PATH 
--windowsize 1600 --stridesize 500 \
--batch_size 256 \
--num_epochs 300 \
--output_class_dim 50 \
--listener_layer 4 \
--comet_api_key ""\
--project_name "" \
--workspace "" 
```




where the arguments represent:
* `val_fold` - which fold for conducting cross validation
* `dataset_name` - the name of the dataset stroke/breakfast/50salads,etc
* `model_name` - the name for this experiment
* `model_path` - the path to save the model
* `path_root` - the path of the dataset
* `windowsize` - window size of the input
* `stridesize` - stride size when dividing the sequence to windows
* `batch_size` - batch size 
* `num_epochs` - number of epochs for training 
* `output_class_dim` - the output number of classes
* `listener_layer` - the number of layers for encoder
* `comet_api_key` - comet api_key
* `project_name` - comet project_name
* `workspace` - comet workspace
