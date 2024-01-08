import numpy as np

from utils import *
from prophets import *


# def Scenario_1(train_set):
#     """
#     Question 1.
#     2 Prophets 1 Game.
#     You may change the input & output parameters of the function as you wish.
#     """
#     prophet_1 = Prophet(0.2)
#     prophet_2 = Prophet(0.4)
#     prophets = [prophet_1, prophet_2]
#     experiments = 100
#     preds = np.zeros(experiments)
#     error = np.zeros(len(prophets))
#     gt = np.zeros(experiments)
#     for experiment in range(experiments):
#
#         "choose random game from the training set"
#         gt[experiment] = np.random.choice(train_set[experiment])
#
#         for prophet in range(len(prophets)):
#             "simulate the predictions that each prophet will make on each training example"
#             preds[experiment] = prophet_1.predict(gt[experiment])
#
#     "estimate the empirical risk on the random game"
#     error[prophet] = compute_error(preds, gt)
#
#     best_prophet = np.argmin(error)
#     avg_error = np.min(error) / 1000
#     print(f"Prophet {best_prophet} is better with an average error of {avg_error}")


def Scenario_1(train_set, test_set):
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    experiments = 100
    num_games_per_set = 1

    prophet_1 = Prophet(0.2)
    prophet_2 = Prophet(0.4)

    generalization_errors = np.zeros(experiments)
    estimation_errors = np.zeros(experiments)
    approximation_errors = np.zeros(experiments)

    right_choice = 0

    for experiment in range(experiments):
        # draw random training and test examples
        gt = np.array([np.random.choice(train_set[experiment], size=num_games_per_set)])
        test = np.array([np.random.choice(test_set, size=num_games_per_set)])

        # simulate the predictions that each prophet will make on each training example
        preds_1 = np.array([prophet_1.predict(gt)])
        preds_2 = np.array([prophet_2.predict(gt)])

        # estimate the empirical risk on the random game
        error_rate_1 = compute_error(preds_1, gt)
        error_rate_2 = compute_error(preds_2, gt)

        # choose the prophet that makes the least errors (the ERM prophet)
        erm_prophet = prophet_1 if error_rate_1 < error_rate_2 else prophet_2
        if erm_prophet == prophet_1:
            right_choice += 1

        # evaluate the empirical risk of the ERM prophet on the test set
        erm_preds = erm_prophet.predict(test)
        erm_error_rate = compute_error(erm_preds, test)

        generalization_errors[experiment] = erm_error_rate - min(error_rate_1, error_rate_2)

        approximation_errors[experiment] = min(error_rate_1, error_rate_2) - min(erm_error_rate,
                                                                                 min(error_rate_1, error_rate_2))
        estimation_errors[experiment] = erm_error_rate - min(error_rate_1, error_rate_2)

    print("Average generalization error:", np.mean(generalization_errors))
    print(f"The best prophet was chosen {right_choice} times")
    print("Average approximation error:", np.mean(approximation_errors))
    print("Average estimation error:", np.mean(estimation_errors))


def Scenario_2():
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_3():
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_4():
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_5():
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


def Scenario_6():
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    pass


if __name__ == '__main__':
    np.random.seed(0)  # DO NOT MOVE / REMOVE THIS CODE LINE!

    # train, validation and test splits for Scenario 1-3, 5
    train_set = create_data(100, 1000)
    test_set = create_data(1, 1000)[0]

    print(f'Scenario 1 Results:')
    Scenario_1(train_set, test_set)

    print(f'Scenario 2 Results:')
    Scenario_2()

    print(f'Scenario 3 Results:')
    Scenario_3()

    print(f'Scenario 4 Results:')
    Scenario_4()

    print(f'Scenario 5 Results:')
    Scenario_5()

    print(f'Scenario 6 Results:')
    Scenario_6()
