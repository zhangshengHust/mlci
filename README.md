# MLCI
 
Implementation of MLCI model for the TextVQA task. The TextVQA task involves visual scenes containing texts and require a simultaneous understanding of images, questions, and texts in images to reason answers. Multi-level Complete Interaction (MLCI) model for the TextVQA task via stacking multiple blocks composed of our proposed interaction modules.
 ## 

 ## Dataset
we use textvqa dataset to evaluate our method. You can download this dataset via the [link](https://textvqa.org/).

 ## Train example
 ```sh
$CUDA_VISIBLE_DEVICES=0 python run.py --config options/exp_1_20_1.yaml --is_train True
 ```

 ## evaluation example
```sh
# test set results
$CUDA_VISIBLE_DEVICES=0 python run.py --config options/exp_1_20_1.yaml --is_train False --eval_name test
# validation set results
$CUDA_VISIBLE_DEVICES=0 python run.py --config options/exp_1_20_1.yaml --is_train False --eval_name val
 ```