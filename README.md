# Combining Generative and Discriminative Models for Hybrid Inference
Implementation of [Combining Generative and Discriminative Models for Hybrid Inference](https://arxiv.org/abs/1906.02547) on Python3, Pytorch 1.3

Dependencies: numpy, scipy, filterpy, pandas, pytorch

## Linear dynamics
```python3 exp1_linear.py```

## Non-linear dynamics - Lorenz attractor

```
python3 exp2_lorenz.py --taylor_K 0
python3 exp2_lorenz.py --taylor_K 1
python3 exp2_lorenz.py --taylor_K 2
```
## Cite
```
@article{satorras2019combining,
  title={Combining Generative and Discriminative Models for Hybrid Inference},
  author={Satorras, Victor Garcia and Akata, Zeynep and Welling, Max},
  journal={arXiv preprint arXiv:1906.02547},
  year={2019}
}
```


The Robert Bosch GmbH is acknowledged for financial support


