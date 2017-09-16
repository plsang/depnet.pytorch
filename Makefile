
OUT_DIR=output
MODEL_DIR=$(OUT_DIR)/model

MSCOCO_DATA=$(CLCV_HOME)/resources/data/Microsoft_COCO
MSCOCO_IMAGE_DIR=$(CLCV_HOME)/resources/corpora/Microsoft_COCO/images
GID?=0

train: $(MODEL_DIR)/mscoco2014_dev1_captions_myconceptsv3.pth
$(MODEL_DIR)/mscoco2014_dev1_captions_myconceptsv3.pth:
	CUDA_VISIBLE_DEVICES=$(GID) python train.py $@ \
			     --train_label_file $(MSCOCO_DATA)/mscoco2014_dev1_captions_myconceptsv3.h5 \
			     --val_label_file $(MSCOCO_DATA)/mscoco2014_dev2_captions_myconceptsv3.h5 \
			     --test_label_file $(MSCOCO_DATA)/mscoco2014_val_captions_myconceptsv3.h5 \
			     --train_imageinfo_file $(MSCOCO_DATA)/mscoco2014_dev1_imageinfo.json \
			     --val_imageinfo_file $(MSCOCO_DATA)/mscoco2014_dev2_imageinfo.json \
			     --test_imageinfo_file $(MSCOCO_DATA)/mscoco2014_val_imageinfo.json \
			     --train_image_dir $(MSCOCO_IMAGE_DIR)/train2014 \
			     --val_image_dir $(MSCOCO_IMAGE_DIR)/train2014 \
			     --test_image_dir $(MSCOCO_IMAGE_DIR)/test2014 
