# Data Preparation

After the organization, we expect the directory structure to be the following:


```
├── LoSh
│   ├── a2d_sentences
│       ├── Release
│       ├── text_annotations
│           ├── a2d_annotation_with_instances
│           ├── a2d_annotation.txt
│           ├── a2d_missed_videos.txt
│       ├── a2d_sentences_single_frame_test_annotations.json
│       ├── a2d_sentences_single_frame_train_annotations.json
│       ├── a2d_sentences_test_annotations_in_coco_format.json
│   ├── jhmdb_sentences
│       ├── Rename_Images
│       ├── puppet_mask
│       ├── jhmdb_annotation.txt
│       ├── jhmdb_sentences_samples_metadata.json
│       ├── jhmdb_sentences_gt_annotations_in_coco_format.json
│   ├── refer_youtube_vos
│         ├── meta_expressions
│         ├── train
│               ├── JPEGImages
│               ├── Annotations
│               ├── meta.json
│         ├── valid
│               ├── JPEGImages
│   ├── refer_davis
│         ├── meta_expressions
│         ├── valid
│               ├── JPEGImages
│                     ├── 480p
│               ├── Annotations
│               ├── ImageSets
│               ├── meta.json
...
```

## A2D-Sentences

Follow the instructions and download the dataset from the website [here](https://kgavrilyuk.github.io/publication/actor_action/). 
Then, extract the files. We expect the directory structure to be the following:

```
├── LoSh
│   ├── a2d_sentences
│   │   ├── Release
│   │   ├── text_annotations
│   │   │   ├── a2d_annotation_with_instances
│   │   │   ├── a2d_annotation.txt
│   │   │   ├── a2d_missed_videos.txt
│   │   ├── a2d_sentences_single_frame_test_annotations.json
│   │   ├── a2d_sentences_single_frame_train_annotations.json
│   │   ├── a2d_sentences_test_annotations_in_coco_format.json
```

## JHMDB-Sentences

Follow the instructions and download the dataset from the website [here](https://kgavrilyuk.github.io/publication/actor_action/). 
Then, extract the files. We expect the directory structure to be the following:

```
├── LoSh
│   ├── jhmdb_sentences
│   │   ├── Rename_Images
│   │   ├── puppet_mask
│   │   ├── jhmdb_annotation.txt
│   │   ├── jhmdb_sentences_samples_metadata.json
│   │   ├── jhmdb_sentences_gt_annotations_in_coco_format.json
```

## Refer_YouTube_VOS

Download the dataset from the competition's website [here](https://competitions.codalab.org/competitions/29139#participate-get_data).
Then, extract and organize the file. We expect the directory structure to be the following:

```
├── LoSh
│   ├── refer_youtube_vos
│         ├── meta_expressions
│         ├── train
│               ├── JPEGImages
│               ├── Annotations
│               ├── meta.json
│         ├── valid
│               ├── JPEGImages
```

## Refer_DAVIS17

Download the DAVIS2017 dataset from the [website](https://davischallenge.org/davis2017/code.html). Note that you only need to download the two zip files `DAVIS-2017-Unsupervised-trainval-480p.zip` and `DAVIS-2017_semantics-480p.zip`.
Download the text annotations from the [website](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/video-segmentation/video-object-segmentation-with-language-referring-expressions).
Then, put the zip files in the directory as follows.

```
├── LoSh
│   ├── refer_davis
│   │   ├── DAVIS-2017_semantics-480p.zip
│   │   ├── DAVIS-2017-Unsupervised-trainval-480p.zip
│   │   ├── davis_text_annotations.zip
```

Unzip these zip files.
```
unzip -o davis_text_annotations.zip
unzip -o DAVIS-2017_semantics-480p.zip
unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```

Preprocess the dataset to refer_youtube_vos format. (Make sure you are in the main directory)

```
python tools/data/convert_davis_to_ytvos.py
```

Finally, unzip the file `DAVIS-2017-Unsupervised-trainval-480p.zip` again (since we use `mv` in preprocess for efficiency).

```
unzip -o DAVIS-2017-Unsupervised-trainval-480p.zip
```

## Short query generation

Users can generate the short text expressions according to our paper via Spacy tool. We also refer to this [link](https://drive.google.com/drive/folders/1R3uDAgE-UMusfqzLsfsjUplTYgEQYj3W?usp=sharing) to download the short queries.

