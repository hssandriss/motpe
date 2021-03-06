# Using MO-TPE for neural architecture search

## Dependencies

Please install the dependencies via:
```sh
pip install -r requirements.txt
```

## Features:
- NAS as HPO but caring about two objectives, **maximizing** test set accuracy and **minimizing** network number of parameters.
- Using hierachichal search space for CNN.
- Using early stopping for worst configurations in similar way as successive halving (multifidelity).

## Original code base made available by paper authors:


[Yoshihiko Ozaki](https://github.com/y0z)

[Shuhei Watanabe](https://github.com/nabenabe0928)

[Multiobjective tree-structured parzen estimator for computationally expensive optimization problems](https://dl.acm.org/doi/abs/10.1145/3377930.3389817)
