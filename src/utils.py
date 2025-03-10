import numpy as np
from rpy2.robjects import numpy2ri
import rpy2.robjects.packages as rpackages
import copy

# Activate the automatic conversion between R and numpy objects
numpy2ri.activate()

# Import necessary R packages
stats = rpackages.importr('stats')

def evaluate_random_policy(covariates, scores):
    predictions = np.random.randint(1, scores.shape[1]+1, size=covariates.shape[0])
    res = [scores[i, int(prediction-1)] for i, prediction in enumerate(predictions)]
    # print(scores)
    policy_value = np.mean(res)
    # print(np.mean(scores, axis=0))
    return policy_value

def evaluate_random_regret(covariates, scores):
    # covariates_r = numpy2ri.py2rpy(covariates)
    # predictions = stats.predict(policy, covariates_r) - 1
    best_policy_value = np.mean([max(row) for row in scores])
    return best_policy_value - evaluate_random_policy(covariates, scores)

def compute_mixture_regret_random(mixture_weights, source_data):
    mixture_regret = 0.0
    # print(source_data)
    for source_id in source_data:
        covariates, _, _, true_rewards = copy.deepcopy(source_data[source_id])
        weight = mixture_weights[source_id]
        # print(evaluate_random_policy(covariates, true_rewards))
        if weight > 0:
            mixture_regret += weight * evaluate_random_regret(covariates, true_rewards)
    return mixture_regret

def evaluate_policy(policy, covariates, scores, best=False):
    covariates_r = numpy2ri.py2rpy(covariates)
    predictions = stats.predict(policy, covariates_r) - 1
    # print(predictions.shape)
    # print(scores.shape)
    policy_value = np.mean([scores[i, int(prediction-1)] for i, prediction in enumerate(predictions)])
    if best:
        policy_value = np.mean([max(row) for row in scores])
    return policy_value

def compute_regret(policy, best_policy, covariates, scores):
    policy_value = evaluate_policy(policy, covariates, scores)
    best_policy_value = evaluate_policy(best_policy, covariates, scores, best=True)
    # best_policy_value = np.mean([max(row) for row in scores])
    # print("Best policy value", best_policy_value)
    return best_policy_value - policy_value

def compute_mixture_policy_value(policy, mixture_weights, source_data):
    mixture_policy_value = 0.0
    for source_id in source_data:
        covariates, _, _, true_rewards = copy.deepcopy(source_data[source_id])
        weight = mixture_weights[source_id]
        if weight > 0:
            mixture_policy_value += weight * evaluate_policy(policy, covariates, true_rewards)
    return mixture_policy_value

def compute_mixture_regret(policy, best_policy, mixture_weights, source_data):
    mixture_regret = 0.0
    for source_id in source_data:
        covariates, _, _, true_rewards = copy.deepcopy(source_data[source_id])
        weight = mixture_weights[source_id]
        if weight > 0:
            mixture_regret += weight * compute_regret(policy, best_policy, covariates, true_rewards)
    return mixture_regret
