# Mask RCNN instance segmentation of groceries
I used Jupyter Notebook to save the output of the cells, in case You want to see it. The notebooks were used in the training and visual evaluation part.  
## Training
Training Notebook is [training_maskrcnn.ipynb](https://github.com/macodroid/maskrcnn-groceries/blob/main/training_maskrcnn.ipynb)  
To start training You need first to change the paths of dataset in [config_maskrcnn.py](https://github.com/macodroid/maskrcnn-groceries/blob/main/config_maskrcnn.py) (```train_root_dir, train_ann_file, test_root_dir, test_ann_file```)  
Final validation mAP0.5:0.95 is **0.667**
## Evaluation
Evaluation Notebook is [eval_maskrcnn.ipynb](https://github.com/macodroid/maskrcnn-groceries/blob/main/eval_maskrcnn.ipynb)