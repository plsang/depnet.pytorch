# Dependency Prediction Networks #

This project is for dependency prediction from images.

## Dependencies ##

* Python 2.7
* Pytorch 0.2 
* [clcv project](https://github.com/mynlp/clcv)

## Getting started ##

* This project works with data format produced by the clcv project. In order to start, please set the `CLCV_HOME` evironment variable point to the CLCV project:
`export CLCV_HOME=/path/to/the/clcv/project`
* Now, everything is ready.

## Training ##

* Basic usage
```bash
  usage: train.py [-h] [--train_image_dir TRAIN_IMAGE_DIR]
              [--val_image_dir VAL_IMAGE_DIR] [--finetune FINETUNE]
              [--cnn_type {vgg19,resnet152}] [--batch_size BATCH_SIZE]
              [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]
              [--lr_update LR_UPDATE] [--max_patience MAX_PATIENCE]
              [--val_step VAL_STEP] [--num_workers NUM_WORKERS]
              [--log_step LOG_STEP]
              [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}] [--seed SEED]
              train_label val_label train_imageinfo val_imageinfo
              output_file

  positional arguments:
  train_label           path to the h5 file containing the training labels
                      info
  val_label             path to the h5 file containing the validating labels
                      info
  train_imageinfo       imageinfo contains image path
  val_imageinfo         imageinfo contains image path
  output_file           output model file (*.pth)

  optional arguments:
  -h, --help            show this help message and exit
  --train_image_dir TRAIN_IMAGE_DIR
                      path to training image dir
  --val_image_dir VAL_IMAGE_DIR
                      path to validating image dir
  --finetune FINETUNE   Fine-tune the image encoder.
  --cnn_type {vgg19,resnet152}
                      The CNN used for image encoder (e.g. vgg19, resnet152)
  --batch_size BATCH_SIZE
                      batch size
  --learning_rate LEARNING_RATE
                      learning rate
  --num_epochs NUM_EPOCHS
                      max number of epochs to run the training
  --lr_update LR_UPDATE
                      Number of epochs to update the learning rate.
  --max_patience MAX_PATIENCE
                      max number of epoch to run since the minima is
                      detected -- early stopping
  --val_step VAL_STEP   how often do we check the model (in terms of epoch)
  --num_workers NUM_WORKERS
```

* Example of using make rules for training
```bash
    make train GID=0 BATCH_SIZE=128 LEARNING_RATE=0.0001 CNN_TYPE=vgg19 FINETUNE=False NUM_WORKERS=4
```

## Testing ##

* Basic usage
```bash
  usage: test.py [-h] [--test_image_dir TEST_IMAGE_DIR]
               [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
               [--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
               test_label test_imageinfo model_file output_file

  positional arguments:
  test_label            path to the h5 file containing the testing labels info
  test_imageinfo        imageinfo contains image path
  model_file            path to the model file
  output_file           path to the output file

  optional arguments:
  -h, --help            show this help message and exit
  --test_image_dir TEST_IMAGE_DIR
                        path to the image dir
  --batch_size BATCH_SIZE
                        batch size
  --num_workers NUM_WORKERS
                        number of workers (each worker use a process to load a
                        batch of data)
  --loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}

```

* Example of using make rules for testing
```bash
    make test GID=0 BATCH_SIZE=128 NUM_WORKERS=4
```

## Results ##
* Train, val, test set are `dev1`, `dev2`, `val` irrespectively
* Performance of the Vgg19 and ResNet152 models (with and without using finetuning) are reported below.

|      Run            |myconceptsv3|mydepsv4|mydepsprepv4|mypasv4|mypasprepv4|exconceptsv3|exdepsv4|exdepsprepv4|expasv4|expasprepv4|
|---------------------|:----------:|:------:|:----------:|:-----:|:---------:|:----------:|:------:|:----------:|:-----:|:---------:|
|Vgg19                |      0.4496|  0.1994|      0.1980| 0.2121|     0.2135|      0.5270|  0.2414|      0.2388| 0.2692|     0.2714|
|ResNet152            |      0.4525|  0.1957|      0.1951| 0.2079|     0.2091|      0.5276|  0.2374|      0.2350| 0.2649|     0.2674|
|Vgg19 +Finetune     |      0.4925|  0.1841|      0.2020| 0.2183|     0.2176|      0.5601|  0.2168|      0.2532| 0.2864|     0.2858|
|ResNet152 +Finetune|      0.5188|  0.2343|      0.2320| 0.2499|     0.2511|      0.5897|  0.2856|      0.2806| 0.3177|     0.3186|

### Some observations ###
* Finetuning is applied at the begining for the whole network. A better strategy would be training the last layer first and then start the finetuning. 
* Without finetuning, performance of Vgg19 and ResNet152 are more or less similar.
* Finetuning can significantly boost the performance, and finetuning on ResNet brings more improvement. 
* Finetuning requires more memory, and the convergence rate is rather slow for the some initial epochs. You may need to increase the `max_patience` hyperparamter to prevent the training from stopping too early, and increase the `num_epochs` to get a better result.
  
## TODO ##

