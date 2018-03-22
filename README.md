# FashionAI


### Training

* Setting the data folder

```bash
$ ln -s /path/to/your/fashionAI/data/folder ./data
```

where your fashionAI data folder should have the tree structure like this:

    /path/to/your/fashionAI/data/folder
    ├── fashionAI_attributes_test_a_20180222
    │   └── rank
    │       ├── Images
    │       └── Tests
    └── fashionAI_attributes_train_20180222
        └── base
            ├── Annotations
            └── Images

* Train the fashionAI model

```bash
$ python fashionAI_main.py --attr_index 0 --model_dir model/skirt
```

where `attr_index` can be
0: skirt_length_labels
1: coat_length_labels
2: collar_design_labels
3: lapel_design_labels
4: neck_design_labels
5: neckline_design_labels
6: pant_length_labels
7: sleeve_length_labels

* Display the training process

```bash
$ tensorboard --logdir model/skirt
```

### Evaluation
