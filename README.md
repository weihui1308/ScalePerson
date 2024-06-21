# The ScalePerson Dataset
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

Download the dataset:

- Google Drive:
[Training Set](https://drive.google.com/file/d/1d98YsPT3a8jpnOBEG123GRFHlRogiDCv/view?usp=sharing) and [Validation Set](https://drive.google.com/file/d/1am_zjTd53L47rPlvR4us43KV25gwN4F6/view?usp=sharing).
- Baidu Netdisk:
[Training Set](https://pan.baidu.com/s/1ZPFjExOgLM2x5Bv29Cta8w?pwd=7384) and [Validation Set](https://pan.baidu.com/s/1Mq-vz8k-yjTZ_j_X79JXGA?pwd=4hxm).

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
