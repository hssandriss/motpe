# Using MO-TPE for neural architecture search

## Dependencies

Please install the dependencies via:
```sh
pip install -r requirements.txt
```

## Features:
- One way of doing NAS as HPO but in a multiobjective optimization, maximizing test set accuracy and minimizing network size.
- Using hierachichal search space.
- Using early stopping for worst configurations in similar way as successive halving (multifidelity).

## Original code base made available by paper authors:


[Yoshihiko Ozaki](https://github.com/y0z)

[Shuhei Watanabe](https://github.com/nabenabe0928)

[Multiobjective tree-structured parzen estimator for computationally expensive optimization problems](https://dl.acm.org/doi/abs/10.1145/3377930.3389817)
