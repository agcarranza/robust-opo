import numpy as np
from tqdm import tqdm
import random
from rpy2.robjects import r, numpy2ri
import rpy2.robjects.packages as rpackages
import rpy2.robjects.vectors as rvectors
import logging
from src import utils
import copy

# Activate the automatic conversion between R and numpy objects
numpy2ri.activate()

# Import necessary R packages
grf = rpackages.importr('grf')
stats = rpackages.importr('stats')
policytree = rpackages.importr('policytree')

class EGPolicyTree:
    def __init__(self, cover, learning_rate, depth=2, epsilon=1e-8, regularization_lambda=0.001): # cover = {expert_id: mixture_weights}
        # Input parameters
        self.cover = cover
        self.num_experts = len(cover)

        # Hyperparameters
        self.depth = depth
        self.learning_rate = learning_rate

        # Trainable parameters
        self.experts_distro = np.ones(self.num_experts) / self.num_experts
        self.trained_policy = None

        # Reward parameters
        self.regularization_lambda = regularization_lambda

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def __update_learning_rate(self, iteration):
        self.current_learning_rate = self.learning_rate / (1 + iteration * 0.1)

    def __subsample_data(self, data, sample_size):
        subsampled_data = {}
        for source_id, dataset in data.items():
            num_samples = dataset[0].shape[0]
            if num_samples < sample_size:
                raise ValueError(f"Sample size {sample_size} is larger than the dataset size for source_id {source_id}")
            
            indices = np.array(random.sample(range(num_samples), sample_size))
            # print(indices)
            subsampled_dataset = []
            for arr in dataset:
                arr_sub = np.array([arr[i] for i in indices])
                subsampled_dataset.append(arr_sub)
            subsampled_data[source_id] = subsampled_dataset

        return subsampled_data

    def fit(self, data, num_iterations, num_subsamples=None, use_true_rewards=False):
        print("Subsampling data...")
        subsampled_data = self.__subsample_data(data, num_subsamples)
        
        print("Computing source scores...")
        if use_true_rewards:
            source_scores = {
                source_id: {
                    'scores': copy.deepcopy(subsampled_data[source_id][-1]),
                    'covariates': copy.deepcopy(subsampled_data[source_id][0])
                }
                for source_id in subsampled_data
            }
        else:
            source_scores = self.__compute_source_scores(subsampled_data)

        print("Computing max policy values...")
        expert_max_policy_values = self.__compute_max_policy_values(source_scores)
        
        print("Training the model...")
        for iteration in tqdm(range(num_iterations)):
            best_response_policy = self.__get_best_response_policy(source_scores)

            expert_policy_value = utils.compute_mixture_regret_random(
                self.experts_distro, subsampled_data
            )
            self.logger.info(f"Iteration {iteration}: Policy value: {expert_policy_value}")

            reward_gradient = self.__compute_reward_gradient(source_scores, best_response_policy, expert_max_policy_values)
            self.logger.info(f"Iteration {iteration}: Gradients: {reward_gradient}")
            self.logger.info(f"Iteration {iteration}: Expert distribution before update: {self.experts_distro}")
            self.__update_learning_rate(iteration)
            self.__update_experts_distribution(reward_gradient)
            self.logger.info(f"Iteration {iteration}: Expert distribution after update: {self.experts_distro}")
            self.trained_policy = best_response_policy

        return
    
    def predict(self, covariates):
        if self.trained_policy is None:
            raise ValueError("Model has not been trained yet.")
        
        covariates_r = numpy2ri.py2rpy(covariates)
        return stats.predict(self.trained_policy, covariates_r).astype(int)

    def __evaluate_policy(self, policy, covariates, scores):
        covariates_r = numpy2ri.py2rpy(covariates)
        predictions = stats.predict(policy, covariates_r) - 1
        policy_value = np.mean([scores[i, int(prediction-1)] for i, prediction in enumerate(predictions)])
        return policy_value
    
    def __compute_source_scores(self, data):
        source_scores = {}
        for source_id in data:
            # Convert numpy arrays to R objects.
            covariates, rewards, treatment, _ = data[source_id]
            covariates_r = numpy2ri.py2rpy(covariates)          # ns x p
            rewards_r = rvectors.FloatVector(-rewards)           # ns x 1
            treatment_r = rvectors.FactorVector(treatment)      # ns x 1

            # Create and train a multi-arm causal forest in R.
            multiforest = grf.multi_arm_causal_forest(covariates_r, rewards_r, treatment_r)
        
            # Compute doubly robust reward estimates.
            scores = np.array(policytree.double_robust_scores(multiforest)) # ns x d
            source_scores[source_id] = {'scores': copy.deepcopy(scores), 'covariates': copy.deepcopy(covariates)}

        return source_scores # {source_id: {'covariates': covariates, 'scores': scores}
    
    def __compute_max_policy_values(self, source_scores):
        max_policy_values = []
        n = sum([source_scores[source_id]['covariates'].shape[0] for source_id in source_scores])
        # print(self.num_experts)
        for expert_id in range(self.num_experts):
            expert_weights = self.cover[expert_id]

            expert_scores = []
            expert_covariates = []
            for source_id in source_scores:
                scores = copy.deepcopy(source_scores[source_id]['scores'])
                covariates = copy.deepcopy(source_scores[source_id]['covariates'])

                # Apply source-specific weight to scores.
                # weight = n * expert_weights[source_id] / covariates.shape[0]
                weight = expert_weights[source_id]
                if weight == 0:
                    continue
                scores *= weight

                # Append to list of scores and covariates.
                expert_scores.append(scores)
                expert_covariates.append(covariates)

            # Stack scores and covariates.
            if len(expert_scores) > 1:
                expert_scores = np.vstack(expert_scores)
                expert_covariates = np.vstack(expert_covariates)
            else:
                expert_scores = copy.deepcopy(expert_scores[0])
                expert_covariates = copy.deepcopy(expert_covariates[0])

            # Train a policy tree on the expert-specific scores and covariates.
            expert_scores_r = numpy2ri.py2rpy(expert_scores)
            expert_covariates_r = numpy2ri.py2rpy(expert_covariates)

            # print(expert_scores_r)
            print(f"for expert {expert_id}...")
            policy = policytree.policy_tree(
                expert_covariates_r, expert_scores_r, depth=self.depth
            )

            # Compute the expert policy value of optimal policy.
            expert_policy_value = 0.0
            for source_id in source_scores:
                scores = copy.deepcopy(source_scores[source_id]['scores'])
                covariates = copy.deepcopy(source_scores[source_id]['covariates'])
                source_policy_value = self.__evaluate_policy(policy, covariates, scores)
                expert_policy_value += expert_weights[source_id] * source_policy_value

            max_policy_values.append(expert_policy_value)

        return np.array(max_policy_values)

    def __get_best_response_policy(self, source_scores):
        all_scores = []
        all_covariates = []

        n = sum([source_scores[source_id]['covariates'].shape[0] for source_id in source_scores])

        for expert_id in range(self.num_experts):
            expert_weights = self.cover[expert_id]
            expert_probability = self.experts_distro[expert_id]

            for source_id in source_scores:
                scores = copy.deepcopy(source_scores[source_id]['scores'])
                covariates = copy.deepcopy(source_scores[source_id]['covariates'])

                # Apply (expert, source)-specific weight to scores.
                # weight = (self.num_experts * n) * expert_probability * expert_weights[source_id] / covariates.shape[0]
                weight = expert_probability * expert_weights[source_id]
                if weight == 0:
                    continue
                scores *= weight

                # Append to list of scores and covariates.
                all_scores.append(scores)
                all_covariates.append(covariates)
    
        # Stack all scores and covariates.
        if len(all_scores) > 1:
            all_scores = np.vstack(all_scores)
            all_covariates = np.vstack(all_covariates)
        else:
            all_scores = np.array(all_scores[0])
            all_covariates = np.array(all_covariates[0])

        # Train a policy tree on the stacked scores and covariates.
        scores_r = numpy2ri.py2rpy(all_scores)
        covariates_r = numpy2ri.py2rpy(all_covariates)
        policy = policytree.policy_tree(covariates_r, scores_r, depth=self.depth)
        self.logger.info(f"Best response policy created with depth {self.depth}")
        return policy
    
    def __compute_reward_gradient(self, source_scores, best_response_policy, expert_max_policy_values):
        reward_gradient = []
        for expert_id in range(self.num_experts):
            expert_weights = self.cover[expert_id]

            # Compute expert-mixture policy value of best response policy.
            expert_policy_value = 0.0
            for source_id in source_scores:
                scores = copy.deepcopy(source_scores[source_id]['scores'])
                covariates = copy.deepcopy(source_scores[source_id]['covariates'])
                source_policy_value = self.__evaluate_policy(best_response_policy, covariates, scores)
                expert_policy_value += expert_weights[source_id] * source_policy_value

            expert_reward_partial_derivative = expert_max_policy_values[expert_id] - expert_policy_value
            reward_gradient.append(expert_reward_partial_derivative)

        return np.array(reward_gradient)

    def __update_experts_distribution(self, gradient):
        self.experts_distro *= np.exp(self.current_learning_rate * np.array(gradient))
        self.experts_distro /= np.sum(self.experts_distro)

        # Regularize the expert distribution    
        self.uniform_distribution = np.ones(self.num_experts) / self.num_experts
        self.experts_distro = (1 - self.regularization_lambda) * self.experts_distro + \
                              self.regularization_lambda * self.uniform_distribution
