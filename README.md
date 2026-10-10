# PathSearch
Official Repository for *Accurate and Scalable Multimodal Pathology Retrieval via Attentive Vision-Language Alignment*

PathSearch is an accurate and scalable system for multimodal pathology retrieval, featuring an attentive mosaic mechanism to boost slide-to-slide retrieval accuracy, while leveraging slide-report alignment to further improve semantic understanding of the slide and enable multimodal retrieval support.

**PathSearch demonstrates higher slide-to-slide retrieval accuracy and faster slide encoding & matching speed than existing frameworks, making it suitable for real-world clinical applications.**

---

> ⚠️ **Note:** The code has been verified for training and inference. If you still find certain files missing, please raise an issue for it. We will continue to ensure that the code behaves the same as in our experiments.

### 1. Prerequisites

To preprocess WSIs in a unified style, [EasyMIL Toolbox](https://github.com/birkhoffkiki/EasyMIL) is highly recommended.  
To process `.kfb`, `.sdpc` format slides in Python, please use the [ASlide](https://github.com/MrPeterJin/ASlide) library.

You will need the following libraries to reproduce or deploy PathSearch (tested on Python 3.9.19):

- **torch** 2.4.0  
- **timm** 0.9.8 (switch to the modified version 0.5.4 for CTransPath/CHIEF, provided in EasyMIL)  
- **einops** 0.8.0  
- **numpy** 1.25.1  
- **scipy** 1.13.1  
- **scikit-learn** 1.6.1  
- **pandas**

The complete experimental environment will be included in the `requirements.txt` file. However, not all libraries listed there are required by PathSearch. The installation time varies between different devices but normally would not takes more than 15 minutes.

### 2. Prepare the data / archive

You can download the TCGA data and corresponding labels from the [NIH Genomic Data Commons](https://portal.gdc.cancer.gov), of which the detailed list is provided in `PathSearch/dataset/TCGA_file_list.txt.`
The Camelyon16 and Camelyon17 datasets are available on the [Grand Challenge](https://camelyon16.grand-challenge.org/) and [Camelyon17](https://camelyon17.grand-challenge.org/) platforms.  
The DHMC-LUAD dataset can be obtained from the Department of Pathology and Laboratory Medicine at Dartmouth–Hitchcock Medical Center via registration and request ([link](https://bmirds.github.io/LungCancer/)). You can also prepare your own datasets as long as you have the whole slide images available.

You may continuously add different types of samples to your search archive, building your own diagnostic library.

#### 2.1 The Demo

We have provided several demo slides for retrieval testing. You can use TCGA-A8-A08L as the query, and PathSearch will return related cases, namely A8-A07C, A8-A079, A8-A0A9, A8-A08P, A8-A09Z.

### 3. Clone the code 

Clone the repository by running:

```
git clone git@github.com:Dootmaan/PathSearch.git
```

Then navigate into the project directory:

```
cd PathSearch
```

### 4. Training

Generally speaking, you can directly use the released weights for the attentive mosaic generator and the report encoder in the PathSearch framework.  
These weights can be found [on Zenodo](https://zenodo.org/records/17431804).

To train PathSearch with the TCGA data pairs, simply run:

```bash
bash shell/train_pathsearch.sh
```

to train the model from scratch with the default hyperparameters.

### 5. Testing

This repository provides four ready-to-run scripts for the four public datasets used in the study, three of which are external. Simply run:

```
bash shell/test.sh
```

to test the model on these datasets. Be sure to specify the path to your archive.

> Note: During testing, cache file will be automatically generated to boost future use. You may need to refresh these cache files manually after making modifications to the pipeline.

#### Acknowledgment

We used [CONCH](https://github.com/mahmoodlab/CONCH) for generating patch-level embeddings via EasyMIL.
We have partially borrowed code from [CLIP](https://github.com/openai/CLIP) and [TransMIL](https://github.com/szc19990412/TransMIL) to construct PathSearch; therefore, PathSearch will also follow the GPL v3 LICENSE upon publication.

We sincerely thank these teams for their dedicated efforts in advancing this field. We also would like to thank the authors from the [PathologySearchComparison project](https://github.com/jacobluber/PathologySearchComparison) for the PyTorch reproduction of existing methods.


#### Citation

If you find this work helpful in your research, please consider citing:
```
@misc{wang2025accuratescalablemultimodalpathology,
      title={Accurate and Scalable Multimodal Pathology Retrieval via Attentive Vision-Language Alignment}, 
      author={Hongyi Wang and Zhengjie Zhu and Jiabo Ma and Fang Wang and Yue Shi and Bo Luo and Jili Wang and Qiuyu Cai and Xiuming Zhang and Yen-Wei Chen and Lanfen Lin and Hao Chen},
      year={2025},
      eprint={2510.23224},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.23224}, 
}
```