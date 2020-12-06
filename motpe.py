from sys import path
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
from numpy.distutils.misc_util import Configuration
from numpy.lib.npyio import load, save
import optproblems.wfg
import pandas as pd
import pyDOE2
from pygmo import hypervolume
import scipy.special
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder
import logging
from cnn import torchModel
from sklearn.model_selection import StratifiedKFold
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
import os

eps = sys.float_info.epsilon

###############################################################################
# benchmark
###############################################################################


class Problem:
    def __init__(self,
                 objective,
                 configspace,
                 n_objectives=1,
                 n_variables=None,
                 hyperparameters=None):
        self.objective = objective
        self.n_objectives = n_objectives
        self.n_variables = len(configspace.get_hyperparameters())
        self.configspace = configspace
        # Memory
        self.memory = []

    def __call__(self, x, budget=20, load=False, save=False, trial=None):
        return self.objective(x, budget, load, save, trial)

    def num_objectives(self):
        return self.objective.num_objectives()


def create_hyperparameter(hp_type,
                          name,
                          lower=None,
                          upper=None,
                          default_value=None,
                          log=False,
                          q=None,
                          choices=None):
    if hp_type == 'int':
        return CSH.UniformIntegerHyperparameter(
            name=name, lower=lower, upper=upper, default_value=default_value, log=log, q=q)
    elif hp_type == 'float':
        return CSH.UniformFloatHyperparameter(
            name=name, lower=lower, upper=upper, default_value=default_value, log=log, q=q)
    elif hp_type == 'cat':
        return CSH.CategoricalHyperparameter(
            name=name, default_value=default_value, choices=choices)
    else:
        raise ValueError('The hp_type must be chosen from [int, float, cat]')


def get_optimizer_and_crit(cfg):
    if 'optimizer' in cfg:
        if cfg['optimizer'] == 'AdamW':
            model_optimizer = torch.optim.AdamW
        else:
            model_optimizer = torch.optim.Adam
    else:
        model_optimizer = torch.optim.Adam

    if 'train_criterion' in cfg:
        if cfg['train_criterion'] == 'mse':
            train_criterion = torch.nn.MSELoss
        else:
            train_criterion = torch.nn.CrossEntropyLoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss
    return model_optimizer, train_criterion


class CnnFromCfg:
    def __init__(self, name: str, seed: int) -> None:
        self.name = name
        self.seed = seed

    def __call__(self, cfg: dict, budget: int, load: bool, save: bool, trial=str):
        """
       Creates an instance of the torch_model and fits the given data on it.
       This is the function-call we try to optimize. Chosen values are stored in
       the configuration (cfg).

       Parameters
       ----------
       cfg: Configuration (basically a dictionary)
           configuration chosen by smac
       seed: int or RandomState
           used to initialize the rf's random generator

       Returns (f0: 1- acc,f1: log10(n_params))
       -------
        """
        print(cfg)
        lr = cfg['learning_rate_init'] if cfg['learning_rate_init'] else 0.001
        # batch_size = cfg['batch_size'] if cfg['batch_size'] else 200
        # Device configuration
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img_width = 16
        img_height = 16
        data_augmentations = transforms.ToTensor()

        data = ImageFolder('micro17flower', transform=data_augmentations)
        targets = data.targets

        # image size
        input_shape = (3, img_width, img_height)

        model = torchModel(cfg,
                           input_shape=input_shape,
                           num_classes=len(data.classes)).to(device)
        total_model_params = np.sum(p.numel() for p in model.parameters())

        if load:
            paths = os.listdir("./models/trial_{}/".format(trial))
            paths.sort(reverse=True)
            load_path = os.path.join("./models/trial_{}/".format(trial), paths[0])
            model.load_state_dict(torch.load(load_path))

        # instantiate optimizer
        model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
        optimizer = model_optimizer(model.parameters(),
                                    lr=lr)
        # instantiate training criterion
        train_criterion = train_criterion().to(device)

        logging.info('Generated Network:')
        summary(model, input_shape,
                device='cuda' if torch.cuda.is_available() else 'cpu')

        # Number of epochs
        # num_epochs = 20
        batch_size = 16
        # Train the model
        score = []

        # returns the cross validation accuracy
        cv = StratifiedKFold(n_splits=3, random_state=self.seed, shuffle=True)  # to make CV splits consistent
        for train_idx, valid_idx in cv.split(data, data.targets):
            train_data = Subset(data, train_idx)
            test_dataset = Subset(data, valid_idx)
            train_loader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=True)
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
            for epoch in range(budget):
                logging.info('#' * 50)
                logging.info('Epoch [{}/{}]'.format(epoch + 1, budget))
                # print(logging.info('Epoch [{}/{}]'.format(epoch + 1, budget)))
                train_score, train_loss = model.train_fn(optimizer, train_criterion, train_loader, device)
                test_score = model.eval_fn(test_loader, device)
                logging.info('Train accuracy %f', train_score)
                logging.info('Test accuracy %f', test_score)
            score.append(test_score)
            # reset model, optimizer and crit
            model = torchModel(cfg,
                               input_shape=input_shape,
                               num_classes=len(data.classes)).to(device)
            total_model_params = np.sum(p.numel() for p in model.parameters())
            # instantiate optimizer
            model_optimizer, train_criterion = get_optimizer_and_crit(cfg)
            optimizer = model_optimizer(model.parameters(),
                                        lr=lr)
            # instantiate training criterion
            train_criterion = train_criterion().to(device)

        fitness = np.array([1 - np.mean(score), np.log10(total_model_params)])
        if save:
            if "trial_{}".format(trial) not in os.listdir("./models"):
                os.mkdir("./models/trial_{}".format(trial))
            save_path = "./models/trial_{}/weights_trial_{}_budget_{}.pt".format(trial, trial, budget)
            torch.save(model.state_dict(), save_path)

        return {f'f{i+1}': fitness[i] for i in range(len(fitness))}  # Because minimize!

    def make_cs(self, cs):
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types,
        # we use continuous, integer and categorical parameters.
        # We can add multiple hyperparameters at once:
        # batch_size = UniformIntegerHyperparameter("batch_size", 32, 256, default_value=64)
        n_conf_layer = UniformIntegerHyperparameter("n_conv_layers", 1, 3, default_value=3)
        n_conf_layer_0 = UniformIntegerHyperparameter("n_channels_conv_0", 16, 256, default_value=256)
        n_conf_layer_1 = UniformIntegerHyperparameter("n_channels_conv_1", 16, 256, default_value=256)
        n_conf_layer_2 = UniformIntegerHyperparameter("n_channels_conv_2", 16, 256, default_value=256)
        # data_dir = Constant('data_dir', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'micro17flower'))
        # num_epochs = Constant('num_epochs', 20)
        learning_rate_init = UniformFloatHyperparameter('learning_rate_init',
                                                        0.00001, 1.0, default_value=2.244958736283895e-05, log=True)
        cs.add_hyperparameters([n_conf_layer, n_conf_layer_0, n_conf_layer_1, n_conf_layer_2,
                                learning_rate_init])

        # Add conditions to restrict the hyperparameter space
        use_conf_layer_2 = CS.conditions.InCondition(n_conf_layer_2, n_conf_layer, [3])
        use_conf_layer_1 = CS.conditions.InCondition(n_conf_layer_1, n_conf_layer, [2, 3])
        # Add  multiple conditions on hyperparameters at once:
        cs.add_conditions([use_conf_layer_2, use_conf_layer_1])
        return cs

    def num_objectives(self):
        return 2


class WFG:
    def __init__(self, name, base_configuration):
        self.name = name
        self.base_configuration = base_configuration

    def __call__(self, configuration):
        num_variables = self.base_configuration['num_variables']
        num_objectives = self.base_configuration['num_objectives']
        k = self.base_configuration['k']
        function = {
            "WFG4": optproblems.wfg.WFG4,
            "WFG5": optproblems.wfg.WFG5,
            "WFG6": optproblems.wfg.WFG6,
            "WFG7": optproblems.wfg.WFG7,
            "WFG8": optproblems.wfg.WFG8,
            "WFG9": optproblems.wfg.WFG9,
        }

        arg = tuple(configuration["x" + str(i)] for i in range(1, num_variables + 1))
        f = function[self.name](num_objectives, num_variables, k)
        fitness = np.array(f.objective_function(arg))

        if 'sigma' in self.base_configuration:
            fitness += self.random_state.normal(0, self.base_configuration['sigma'], len(fitness))
        return {f'f{i+1}': fitness[i] for i in range(len(fitness))}

    def make_cs(self, cs):
        for i in range(1, self.base_configuration['num_variables'] + 1):
            var_name = "x" + str(i)
            hp = create_hyperparameter(
                hp_type="float",
                name=var_name,
                lower=0.0,
                upper=2.0 * i,
                default_value=0.0,
                log=False)
            cs.add_hyperparameter(hp)
        return cs

    def num_objectives(self):
        return self.base_configuration['num_objectives']


###############################################################################
# MOTPE
###############################################################################

def nondominated_sort(points):
    points = points.copy()
    ranks = np.zeros(len(points))
    r = 0
    c = len(points)
    while c > 0:
        extended = np.tile(points, (points.shape[0], 1, 1))
        dominance = np.sum(np.logical_and(
            np.all(extended <= np.swapaxes(extended, 0, 1), axis=2),
            np.any(extended < np.swapaxes(extended, 0, 1), axis=2)), axis=1)
        points[dominance == 0] = 1e9  # mark as used
        ranks[dominance == 0] = r
        r += 1
        c -= np.sum(dominance == 0)
    return ranks


class GammaFunction:
    def __init__(self, gamma=0.10):
        self.gamma = gamma

    def __call__(self, x):
        return int(np.floor(self.gamma * x))  # without upper bound for the number of lower samples


def default_weights(x):
    # default is uniform weights
    # we empirically confirmed that the recency weighting heuristic adopted in
    # Bergstra et al. (2013) seriously degrades performance in multiobjective optimization
    if x == 0:
        return np.asarray([])
    else:
        return np.ones(x)


class GaussKernel:
    def __init__(self, mu, sigma, lb, ub, q):
        self.mu = mu
        self.sigma = max(sigma, eps)
        self.lb, self.ub, self.q = lb, ub, q
        self.norm_const = 1.  # do not delete! this line is needed
        self.norm_const = 1. / (self.cdf(ub) - self.cdf(lb))

    def pdf(self, x):
        if self.q is None:
            z = 2.50662827 * self.sigma  # np.sqrt(2 * np.pi) * self.sigma
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return self.norm_const / z * np.exp(-0.5 * mahalanobis)
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            return np.maximum(integral_u - integral_l, eps)

    def log_pdf(self, x):
        if self.q is None:
            z = 2.50662827 * self.sigma  # np.sqrt(2 * np.pi) * self.sigma
            mahalanobis = ((x - self.mu) / self.sigma) ** 2
            return np.log(self.norm_const / z) - 0.5 * mahalanobis
        else:
            integral_u = self.cdf(np.minimum(x + 0.5 * self.q, self.ub))
            integral_l = self.cdf(np.maximum(x + 0.5 * self.q, self.lb))
            return np.log(np.maximum(integral_u - integral_l, eps))

    def cdf(self, x):
        z = (x - self.mu) / (1.41421356 * self.sigma)  # (x - self.mu) / (np.sqrt(2) * self.sigma)
        return np.maximum(self.norm_const * 0.5 * (1. + scipy.special.erf(z)), eps)

    def sample_from_kernel(self, rng):
        while True:
            sample = rng.normal(loc=self.mu, scale=self.sigma)
            if self.lb <= sample <= self.ub:
                return sample


class AitchisonAitkenKernel:
    def __init__(self, choice, n_choices, top=0.9):
        self.n_choices = n_choices
        self.choice = choice
        self.top = top

    def cdf(self, x):
        if x == self.choice:
            return self.top
        elif 0 <= x <= self.n_choices - 1:
            return (1. - self.top) / (self.n_choices - 1)
        else:
            raise ValueError('The choice must be between {} and {}, but {} was given.'.format(
                0, self.n_choices - 1, x))

    def log_cdf(self, x):
        return np.log(self.cdf(x))

    def cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.cdf(x))
        return return_val

    def log_cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.log_cdf(x))
        return return_val

    def probabilities(self):
        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        choice_one_hot = rng.multinomial(n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))[0]


class UniformKernel:
    def __init__(self, n_choices):
        self.n_choices = n_choices

    def cdf(self, x):
        if 0 <= x <= self.n_choices - 1:
            return 1. / self.n_choices
        else:
            raise ValueError('The choice must be between {} and {}, but {} was given.'.format(
                0, self.n_choices - 1, x))

    def log_cdf(self, x):
        return np.log(self.cdf(x))

    def cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.cdf(x))
        return return_val

    def log_cdf_for_numpy(self, xs):
        return_val = np.array([])
        for x in xs:
            return_val = np.append(return_val, self.log_cdf(x))
        return return_val

    def probabilities(self):
        return np.array([self.cdf(n) for n in range(self.n_choices)])

    def sample_from_kernel(self, rng):
        choice_one_hot = rng.multinomial(n=1, pvals=self.probabilities(), size=1)
        return np.dot(choice_one_hot, np.arange(self.n_choices))[0]


class NumericalParzenEstimator:
    def __init__(self, samples, lb, ub, weights_func, q=None, rule='james'):
        self.lb, self.ub, self.q, self.rule = lb, ub, q, rule
        self.weights, self.mus, self.sigmas = self._calculate(samples, weights_func)
        self.basis = [GaussKernel(m, s, lb, ub, q) for m, s in zip(self.mus, self.sigmas)]

    def sample_from_density_estimator(self, rng, n_ei_candidates):
        samples = np.asarray([], dtype=float)
        while samples.size < n_ei_candidates:
            active = np.argmax(rng.multinomial(1, self.weights))
            drawn_hp = self.basis[active].sample_from_kernel(rng)
            samples = np.append(samples, drawn_hp)

        return samples if self.q is None else np.round(samples / self.q) * self.q

    def log_likelihood(self, xs):
        ps = np.zeros(xs.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.pdf(xs)

        return np.log(ps + eps)

    def basis_loglikelihood(self, xs):
        return_vals = np.zeros((len(self.basis), xs.size), dtype=float)
        for basis_idx, b in enumerate(self.basis):
            return_vals[basis_idx] += b.log_pdf(xs)

        return return_vals

    def _calculate(self, samples, weights_func):
        if self.rule == 'james':
            return self._calculate_by_james_rule(samples, weights_func)
        else:
            raise ValueError('unknown rule')

    def _calculate_by_james_rule(self, samples, weights_func):
        mus = np.append(samples, 0.5 * (self.lb + self.ub))
        sigma_bounds = [(self.ub - self.lb) / min(100.0, mus.size), self.ub - self.lb]

        order = np.argsort(mus)
        sorted_mus = mus[order]
        original_order = np.arange(mus.size)[order]
        prior_pos = np.where(original_order == mus.size - 1)[0][0]

        sorted_mus_with_bounds = np.insert([sorted_mus[0], sorted_mus[-1]], 1, sorted_mus)
        sigmas = np.maximum(sorted_mus_with_bounds[1:-1] - sorted_mus_with_bounds[0:-2],
                            sorted_mus_with_bounds[2:] - sorted_mus_with_bounds[1:-1])
        sigmas = np.clip(sigmas, sigma_bounds[0], sigma_bounds[1])
        sigmas[prior_pos] = sigma_bounds[1]

        weights = weights_func(mus.size)
        weights /= weights.sum()

        return weights, mus, sigmas[original_order]


class CategoricalParzenEstimator:
    # note: this implementation has not been verified yet
    def __init__(self, samples, n_choices, weights_func, top=0.9):
        self.n_choices = n_choices
        self.mus = samples
        self.basis = [AitchisonAitkenKernel(c, n_choices, top=top) for c in samples]
        self.basis.append(UniformKernel(n_choices))
        self.weights = weights_func(samples.size + 1)
        self.weights /= self.weights.sum()

    def sample_from_density_estimator(self, rng, n_ei_candidates):
        basis_samples = rng.multinomial(n=1, pvals=self.weights, size=n_ei_candidates)
        basis_idxs = np.dot(basis_samples, np.arange(self.weights.size))
        return np.array([self.basis[idx].sample_from_kernel(rng) for idx in basis_idxs])

    def log_likelihood(self, values):
        ps = np.zeros(values.shape, dtype=float)
        for w, b in zip(self.weights, self.basis):
            ps += w * b.cdf_for_numpy(values)
        return np.log(ps + eps)

    def basis_loglikelihood(self, xs):
        return_vals = np.zeros((len(self.basis), xs.size), dtype=float)
        for basis_idx, b in enumerate(self.basis):
            return_vals[basis_idx] += b.log_cdf_for_numpy(xs)
        return return_vals


class TPESampler:
    def __init__(self,
                 hp,
                 observations,
                 random_state,
                 n_ei_candidates=24,
                 rule='james',
                 gamma_func=GammaFunction(),
                 weights_func=default_weights,
                 split_cache=None):
        self.hp = hp
        self._observations = observations
        self._random_state = random_state
        self.n_ei_candidates = n_ei_candidates
        self.gamma_func = gamma_func
        self.weights_func = weights_func
        self.opt = self.sample
        self.rule = rule
        if split_cache:
            self.split_cache = split_cache
        else:
            self.split_cache = dict()

    def sample(self):
        hp_values, ys = self._load_hp_values()
        print(hp_values, ys)
        n_lower = self.gamma_func(len(hp_values))
        print("n lower", n_lower)
        lower_vals, upper_vals = self._split_observations(hp_values, ys, n_lower)
        var_type = self._distribution_type()

        if var_type in [float, int]:
            hp_value = self._sample_numerical(var_type, lower_vals, upper_vals)
        else:
            hp_value = self._sample_categorical(lower_vals, upper_vals)
        return self._revert_hp(hp_value)

    def _split_observations(self, hp_values, ys, n_lower):
        SPLITCACHE_KEY = str(ys)
        if SPLITCACHE_KEY in self.split_cache:
            lower_indices = self.split_cache[SPLITCACHE_KEY]['lower_indices']
            upper_indices = self.split_cache[SPLITCACHE_KEY]['upper_indices']
        else:
            rank = nondominated_sort(ys)
            indices = np.array(range(len(ys)))
            lower_indices = np.array([], dtype=int)
            # print("rank", rank)
            # print("indices", indices)
            # nondominance rank-based selection
            i = 0
            while len(lower_indices) + sum(rank == i) <= n_lower:
                lower_indices = np.append(lower_indices, indices[rank == i])
                i += 1
                # print(len(lower_indices) + sum(rank == i))

            # hypervolume contribution-based selection
            ys_r = ys[rank == i]
            indices_r = indices[rank == i]
            worst_point = np.max(ys, axis=0)
            reference_point = np.maximum(
                np.maximum(
                    1.1 * worst_point,  # case: value > 0
                    0.9 * worst_point  # case: value < 0
                ),
                np.full(len(worst_point), eps)  # case: value = 0
            )

            S = []
            contributions = []
            for j in range(len(ys_r)):
                contributions.append(hypervolume([ys_r[j]]).compute(reference_point))
            while len(lower_indices) + 1 <= n_lower:
                hv_S = 0
                if len(S) > 0:
                    hv_S = hypervolume(S).compute(reference_point)
                index = np.argmax(contributions)
                contributions[index] = -1e9  # mark as already selected
                for j in range(len(contributions)):
                    if j == index:
                        continue
                    p_q = np.max([ys_r[index], ys_r[j]], axis=0)
                    contributions[j] = contributions[j] \
                        - (hypervolume(S + [p_q]).compute(reference_point) - hv_S)
                S = S + [ys_r[index]]
                lower_indices = np.append(lower_indices, indices_r[index])
            upper_indices = np.setdiff1d(indices, lower_indices)

            self.split_cache[SPLITCACHE_KEY] = {
                'lower_indices': lower_indices, 'upper_indices': upper_indices}

        return hp_values[lower_indices], hp_values[upper_indices]

    def _distribution_type(self):
        cs_dist = str(type(self.hp))

        if 'Integer' in cs_dist:
            return int
        elif 'Float' in cs_dist:
            return float
        elif 'Categorical' in cs_dist:
            var_type = type(self.hp.choices[0])
            if var_type == str or var_type == bool:
                return var_type
            else:
                raise ValueError('The type of categorical parameters must be "bool" or "str".')
        else:
            raise NotImplementedError('The distribution is not implemented.')

    def _get_hp_info(self):
        try:
            if not self.hp.log:
                return self.hp.lower, self.hp.upper, self.hp.q, self.hp.log
            else:
                return np.log(self.hp.lower), np.log(self.hp.upper), self.hp.q, self.hp.log
        except NotImplementedError:
            raise NotImplementedError('Categorical parameters do not have the log scale option.')

    def _convert_hp(self, hp_value):
        try:
            lb, ub, _, log = self._get_hp_info()
            hp_value = np.log(hp_value) if log else hp_value
            return (hp_value - lb) / (ub - lb)
        except NotImplementedError:
            raise NotImplementedError('Categorical parameters do not have lower and upper options.')

    def _revert_hp(self, hp_converted_value):
        try:
            lb, ub, q, log = self._get_hp_info()
            var_type = self._distribution_type()
            hp_value = (ub - lb) * hp_converted_value + lb
            hp_value = np.exp(hp_value) if log else hp_value
            hp_value = np.round(hp_value / q) * q if q is not None else hp_value
            return float(hp_value) if var_type is float else int(np.round(hp_value))
        except NotImplementedError:
            raise NotImplementedError('Categorical parameters do not have lower and upper options.')

    def _load_hp_values(self):
        hp_values = np.array([h['x'][self.hp.name]
                              for h in self._observations if self.hp.name in h['x']])
        hp_values = np.array([self._convert_hp(hp_value) for hp_value in hp_values])
        ys = np.array([np.array(list(h['f'].values()))
                       for h in self._observations if self.hp.name in h['x']])
        # order the newest sample first
        hp_values = np.flip(hp_values)
        ys = np.flip(ys, axis=0)
        return hp_values, ys

    def _sample_numerical(self, var_type, lower_vals, upper_vals):
        q, log, lb, ub, converted_q = self.hp.q, self.hp.log, 0., 1., None

        if var_type is int or q is not None:
            if not log:
                converted_q = 1. / (self.hp.upper - self.hp.lower) \
                    if q is None else q / (self.hp.upper - self.hp.lower)
                lb -= 0.5 * converted_q
                ub += 0.5 * converted_q

        pe_lower = NumericalParzenEstimator(
            lower_vals, lb, ub, self.weights_func, q=converted_q, rule=self.rule)
        pe_upper = NumericalParzenEstimator(
            upper_vals, lb, ub, self.weights_func, q=converted_q, rule=self.rule)
        return self._compare_candidates(pe_lower, pe_upper)

    def _sample_categorical(self, lower_vals, upper_vals):
        choices = self.hp.choices
        n_choices = len(choices)
        lower_vals = [choices.index(val) for val in lower_vals]
        upper_vals = [choices.index(val) for val in upper_vals]

        pe_lower = CategoricalParzenEstimator(
            lower_vals, n_choices, self.weights_func)
        pe_upper = CategoricalParzenEstimator(
            upper_vals, n_choices, self.weights_func)

        best_choice_idx = int(self._compare_candidates(pe_lower, pe_upper))
        return choices[best_choice_idx]

    def _compare_candidates(self, pe_lower, pe_upper):
        samples_lower = pe_lower.sample_from_density_estimator(
            self._random_state, self.n_ei_candidates)
        best_idx = np.argmax(
            pe_lower.log_likelihood(samples_lower) - pe_upper.log_likelihood(samples_lower))
        return samples_lower[best_idx]


class MOTPE:
    def __init__(self, seed=None):
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)
        self._history = []

    def solve(self, problem, parameters):
        cs = problem.configspace
        hyperparameters = cs.get_hyperparameters()
        print(hyperparameters)

        n_variables = problem.n_variables
        seed = self.seed
        init_method = parameters['init_method']
        n_init_samples = parameters['num_initial_samples']
        problem.memory.clear()
        i = 0
        # Generating initial configurartions
        if init_method == 'lhs':
            xs = pyDOE2.lhs(n_variables, samples=n_init_samples, criterion='maximin',
                            random_state=self.random_state)
        for _ in range(n_init_samples):
            if init_method == 'random':
                x = cs.sample_configuration().get_dictionary()
            elif init_method == 'lhs':
                # note: do not use lhs for non-real-valued parameters
                x = {d[0].name: (d[0].upper - d[0].lower) * d[1] + d[0].lower
                     for d in zip(hyperparameters, xs[i])}
            else:
                raise Exception('unknown init_method')
            r = problem(x, budget=20, save=True, trial=str(i))
            record = {'Trial': i, 'x': x, 'f': r, 'budget': 20}
            self._history.append(record)
            print(record)
            i += 1

        # todo: implement sampling conditional parameters
        # TODO: successive halving:
        # FIXME!
        # idea: save and reload models but the problem is that new TPE will be updated based on the lower budget
        # Save after this loop
        while len(self._history) < parameters['num_max_evals']:
            split_cache = {}
            x = {}
            skip1 = False
            skip2 = False
            for hp in cs.get_hyperparameters():
                if hp.name == "n_conv_layers":
                    sampler = TPESampler(hp,
                                         self._history,
                                         self.random_state,
                                         n_ei_candidates=parameters['num_candidates'],
                                         gamma_func=GammaFunction(parameters['gamma']),
                                         weights_func=default_weights,
                                         split_cache=split_cache)
                    x[hp.name] = sampler.sample()
                    split_cache = sampler.split_cache
                    break
            if x["n_conv_layers"] == 1:
                skip1 = True
                skip2 = True
            elif x["n_conv_layers"] == 2:
                skip2 = True

            for hp in cs.get_hyperparameters():
                if hp.name == "n_conv_layers":
                    continue
                if hp.name == "n_channels_conv_1" and skip1:
                    continue
                elif hp.name == "n_channels_conv_2" and skip2:
                    continue
                else:
                    # For each HP sample new value by fitting TPE
                    sampler = TPESampler(hp,
                                         self._history,
                                         self.random_state,
                                         n_ei_candidates=parameters['num_candidates'],
                                         gamma_func=GammaFunction(parameters['gamma']),
                                         weights_func=default_weights,
                                         split_cache=split_cache)
                    x[hp.name] = sampler.sample()
                    split_cache = sampler.split_cache
            r = problem(x, budget=20, save=True, trial=str(i))
            record = {'Trial': i, 'x': x, 'f': r, 'budget': 20}
            self._history.append(record)
            problem.memory.append(record)
            i += 1
        # Reload and continue train here for the top accuracies for 30 epochs
        top_ratio = int(np.floor(len(problem.memory)*(20/30)))
        problem.memory.sort(key=lambda x: x['f']['f1'], reverse=True)  # Sort according to acc
        for record in problem.memory[:top_ratio]:
            r = problem(record['x'], budget=30, save=True, load=True, trial=str(record['Trial']))
            new_record = {'Trial': record['Trial'], 'x': record['x'], 'f': r, 'budget': 50}
            self._history.append(new_record)
        return self.history()

    def history(self):
        return pd.DataFrame.from_dict(self._history)
