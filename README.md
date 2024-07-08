# The ScalePerson Dataset
![Figure](https://github.com/weihui1308/ScalePerson/blob/main/assets/datasetDisplay.svg?raw=true)

ScalePerson dataset is the first of its kind explicitly designed to evaluate physical adversarial attacks against person detection systems. It captures images of individuals at varying distances across diverse real-world scenarios, such as campuses, streets, forests, and indoor settings, ensuring a balanced distribution of person instances at each scale. The dataset includes detailed annotations covering a person's orientation, the number of persons in an image, scene type, and imaging device, facilitating a comprehensive quantitative assessment of attack effectiveness across multiple dimensions. This dataset aims to address the issue of uneven person scale distribution in existing datasets, providing a more realistic and challenging testbed for evaluating the impact of person scale on attack performance.

For more information, see our [website](https://scaleperson.github.io/) and our paper: [ScalePerson: Towards Good Practices in Evaluating Physical Adversarial Attacks on Person Detection](https://scaleperson.github.io/)

## :toolbox: Setup
Clone this repo:
```bash
git clone git@github.com:weihui1308/ScalePerson.git
cd ScalePerson
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Test code runtime environment:
```bash
python tools/test_env.py
```

## :floppy_disk: Download
Download the dataset:

- Google Drive:
[Training Set](https://drive.google.com/file/d/1d98YsPT3a8jpnOBEG123GRFHlRogiDCv/view?usp=sharing) and [Validation Set](https://drive.google.com/file/d/1am_zjTd53L47rPlvR4us43KV25gwN4F6/view?usp=sharing).
- Baidu Netdisk:
[Training Set](https://pan.baidu.com/s/1ZPFjExOgLM2x5Bv29Cta8w?pwd=7384) and [Validation Set](https://pan.baidu.com/s/1Mq-vz8k-yjTZ_j_X79JXGA?pwd=4hxm).

Dataset licence:

The dataset is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## :trophy: Evaluation
1. Add adversarial perturbations, such as adversarial patches, to the dataset.
```bash
python add_patch_toDataset.py --image_path "/home/dataset/images/val/" --label_path "/home/dataset/labels/val/" --output_dir "/home/dataset/images/val_with_attack/" --ratio_h 0.23 --ratio_w 0.17 --patch_path patches/advTshirt.png
```
ratio_h and ratio_w control the height and width of the adversarial patch, respectively.

2. To perform inference on the target dataset and save the results to a JSON file, using Faster R-CNN as an example:
```bash
python inference.py --checkpoint "/home/faster_rcnn_r50.pth" --config dataset/config.py --img_folder "/home/val/" --save_path "runs/fasterRCNN.json"
```
3. Compute the Average Precision (AP) and Attack Success Rate (ASR).
```bash
python eval/calculate_metric.py --json_gt INRIAPerson_coco_annotations.json --json_benign "/home/yolov5s_on_InriaPerson_predictions.json" --json_attack "/home/yolov5s_on_InriaPerson_with_patch_predictions.json" --image_suffix .png
```

<!--
## :pencil2: Citation
If you use this code in your research, please cite our paper:
```
@article{pan2023machiavelli,
    author = {Pan, Alexander and Chan, Jun Shern and Zou, Andy and Li, Nathaniel and Basart, Steven and Woodside, Thomas and Ng, Jonathan and Zhang, Hanlin and Emmons, Scott and Hendrycks, Dan},
    title = {Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the Machiavelli Benchmark.},
    journal = {ICML},
    year = {2023},
}
```
-->
