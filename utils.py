import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import plotly.express as px


# Note: You are not allowed to add any additional imports!

def create_data(n_sets, n_samples):
    """
    Creates a 2-d numpy array of labels.
    y values are randomly selected from {0, 1}
    :param n_sets: number of sets
    :param n_samples: number of points
    :return: y
    """

    y = np.random.choice([0, 1], size=(n_sets, n_samples))
    return y


def compute_error(preds, gt):
    """
    Computes the error of the predictions
    :param preds: predictions
    :param gt: ground truth
    :return: error
    """

    error = np.mean(preds != gt)
    return error
