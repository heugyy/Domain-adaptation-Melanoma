# Domain adaptation for skin cancer classification

## Summary
This work aims at alieviate the domain shift issue in skin cancer classification problem, as shows in the table bellow.

| Dataset name  | Train | Test-HAM  | Test-MoleMap |
| ------------- | ------------- |---------|-------|
| HAM  | 0.952  | 0.907 | 0.310 |
| MoleMap | 0.900 | 0.535 | 0.795 |
| HAM+MoleMap | 0.913 | 0.902 | 0.792 |

We compared classification performance between w/wo applying transfer learning, as well as applying w/wo CycleGAN on target domain. 


Please see this paper for more details: [Progressive transfer learning and adversarial domain adaptation for cross-domain skin disease classification]{https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8846038}.

**Bibtex:**

@article{gu2019progressive,
  title={Progressive transfer learning and adversarial domain adaptation for cross-domain skin disease classification},
  author={Gu, Yanyang and Ge, Zongyuan and Bonnington, C Paul and Zhou, Jun},
  journal={IEEE journal of biomedical and health informatics},
  volume={24},
  number={5},
  pages={1379--1393},
  year={2019},
  publisher={IEEE}
}

## Repo description
[1step-transfer-learning](https://github.com/heugyy/Domain-adaptation-Melanoma/blob/master/1step_transfer_learning.py) is training a classification model using ImageNet pre-train model, and [2step-transfer-learning](https://github.com/heugyy/Domain-adaptation-Melanoma/blob/master/2step_transfer_learning.py) is training a middel model on a larger dataset first before training on the final dataset.



