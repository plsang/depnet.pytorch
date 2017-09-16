
OUT_DIR=output
MODEL_DIR=$(OUT_DIR)/model

MSCOCO_DATA=$(CLCV_HOME)/resources/data/Microsoft_COCO
MSCOCO_IMAGE_DIR=$(CLCV_HOME)/resources/corpora/Microsoft_COCO/images
GID?=0

BATCH_SIZE?=128
LEARNING_RATE?=0.0001
CNN_TYPE?=vgg19
FINETUNE?=False

EXP_NAME?=$(CNN_TYPE)_$(FINETUNE)

MYCONCEPTS=myconceptsv3 mydepsv4 mydepsprepv4 mypasv4 mypasprepv4 
EXCONCEPTS=exconceptsv3 exdepsv4 exdepsprepv4 expasv4 expasprepv4 
SPLITS=dev1 dev2
TRAIN_SPLIT=$(firstword $(SPLITS))

train: $(patsubst %,$(MODEL_DIR)/$(EXP_NAME)/mscoco2014_$(TRAIN_SPLIT)_captions_%.pth,$(MYCONCEPTS) $(EXCONCEPTS))
$(MODEL_DIR)/$(EXP_NAME)/mscoco2014_$(TRAIN_SPLIT)_captions_%.pth: \
       $(foreach s,$(SPLITS), $(MSCOCO_DATA)/mscoco2014_$(s)_captions_%.h5) \
       $(patsubst %,$(MSCOCO_DATA)/mscoco2014_%_imageinfo.json, $(SPLITS))
	mkdir -p $(MODEL_DIR)/$(EXP_NAME)
	CUDA_VISIBLE_DEVICES=$(GID) python train.py $^ $@ \
			     --batch_size $(BATCH_SIZE) --learning_rate $(LEARNING_RATE) \
			     --cnn_type $(CNN_TYPE) --finetune $(FINETUNE) \
			     --train_image_dir $(MSCOCO_IMAGE_DIR)/train2014 \
			     --val_image_dir $(MSCOCO_IMAGE_DIR)/train2014 \
			     2>&1 | tee $(basename $@).log
