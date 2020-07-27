class Gradient_descent:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import datasets

    X, y = datasets.make_regression(n_samples = 100, n_features = 3, n_informative = 1, noise = 30, random_state = 0)

    # gradient_descent(no_features, testSize, randomState, X, y)
    



    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

    from sklearn.linear_model import LinearRegression
    LR = LinearRegression()

    LR.fit(X_train, y_train)

    # print("The values of X_train ---->", X_train)

    constant_list = []
    trained_features = []

    for i in range(3):
        j = "w" + str(i)
        constant_list.append(j)

        feature = X_train[:, i].reshape(len(X_train))
        # feature = X_train[:, i]
        # print(feature)
        trained_features.append(feature)




    epochs = 10000
    lr = 0.001
    c = 0

    parameters = []


    result_dict = {}

    for iteration in range(epochs):

        parameters = []

        if iteration == 0:
            for i in range(3):
                constant = constant_list[i] = 0
                k = constant * trained_features[i]
                parameters.append(k)
        else:

            derivative_variable_values

            for i in range(3):
                constant = derivative_variable_values[i]
                k = constant * trained_features[i]
                parameters.append(k)

        y_pred = c
        for i in parameters:
            y_pred += i

        loss = sum(y_train - y_pred) ** 2 / len(X_train)

        d_c = (-2 / len(X_train)) * sum(y_train - y_pred)   

        c = c - (lr * d_c)
        derivative_variable = []

        for i in range(3):
            i = str(i)

            variable = "d_" + i
            derivative_variable.append(variable)

        print("derivative_variable---->",derivative_variable)
        derivative_variable_values = []
        for j in range(3):
            # print(derivative_variable[1]) 
            # print(trained_features[1])


            value = derivative_variable[j] = (-2 / len(X_train)) * sum((y_train - y_pred) * trained_features[j])
            # print(value)
            value_constants = constant_list[j] = constant_list[j] - (lr * value)

            derivative_variable_values.append(value_constants)


            result_dict[iteration] = loss

        result_dict

      # print("The loss after ", i , "iteration is ", loss)


    result_dict



obj = Gradient_descent()

print(len(obj.result_dict))