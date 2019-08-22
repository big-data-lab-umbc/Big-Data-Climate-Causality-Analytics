import sys
import scipy
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import itertools
from datetime import datetime

startTime = datetime.now()
print("starting time: ", startTime)

maxlag = int(sys.argv[1])
data_file_name = sys.argv[2]
alpha = float(sys.argv[3])

df = pd.read_csv(data_file_name, header='infer')
# n: number of observations
n = df.shape[0]
k = df.shape[1]
print(n)
print(k)
x_list = list(df)
print(x_list)


def regression(df, x_name, y_name, maxlag):
    v = 0.1
    data = df
    df_list = []

    # add lagged columns of current x variable as x_name
    for lag in range(1, maxlag + 1):
        data['{}_{}'.format(x_name, str(lag))] = data['{}'.format(x_name)].shift(lag)
        df_list.append(data['{}_{}'.format(x_name, str(lag))])

    # create test dataframe X, and y
    X = pd.concat(df_list, axis=1)
    y = data[y_name]

    # remove NaN rows, the number of removal is maxlag
    X = X.iloc[maxlag:]
    y = y.iloc[maxlag:]

    # build regression reg_y, X→y
    reg_y = LinearRegression()
    # fit model using data X, y
    reg_y.fit(X, y)

    # y_hat is the predicted value of y
    y_hat = reg_y.predict(X)

    # save predicted y_hat as a pd dataframe and move its index to match the place in original df
    y_hat_df = pd.DataFrame(y_hat)
    y_hat_df.index += maxlag
    # save the predicted value into dataframe
    data['predicted_{}'.format(y_name)] = y_hat_df
    print(data.head(10))
    # compuate mse
    reg_mse = mean_squared_error(y, y_hat)
    # compute residual value of y, y-y_hat, the residual value is the y in next round of loop
    if y_name == x_name:
        # learning rate is not in model 0
        y_residual = y - y_hat
        # apply leraning rate
    else:
        y_residual = y - (y_hat * v)
    data["{}res{}".format(y_name, x_name)] = y_residual

    # print mse, r^2, variance
    print("the mse is")
    print(reg_mse)
    print("regression score is")
    #     print(r2_score(data['{}'.format(y_name)].iloc[3:], data['predicted_{}'.format(y_name)].iloc[3:]))
    # score is the r2_score, same results
    print(reg_y.score(X, y))
    r2 = reg_y.score(X, y)

    # print explained_variance_score
    print("explained_variance_score")
    variance_score = explained_variance_score(y, y_hat)
    print(variance_score)

    return reg_mse, reg_y.score(X, y), variance_score, r2


def boosting(x_list, y_name, maxlag):
    # loop through each variable in the list
    temp_y_name = y_name
    mse_arr = []
    r2_arr = []

    predicted_name_list = []

    for pivot_x in range(0, len(x_list)):
        print("=========this is regression round {}=========".format(pivot_x + 1))

        # save return value of regression in res_list
        res_list = regression(df, x_list[pivot_x], y_name, 3)

        # save predicted column name as a list
        predicted_name_list.append('predicted_{}'.format(y_name))

        # build y_name such as x1resx1, which means x1 substacts x1_hat, res means residual
        y_name = str(y_name) + "res" + str(x_list[pivot_x])

        # example: [0.7614110755692759, 0.6019695603895466, 0.4941602516989991, 0.36284165024184334]
        mse_arr.append(res_list[0])
        r2_arr.append(res_list[3])

    return mse_arr, predicted_name_list, r2_arr, maxlag, x_list


def causality_test(boosting_result_list):
    mse_arr = boosting_result_list[0]
    name_list = boosting_result_list[1]
    r2_arr = boosting_result_list[2]
    maxlag = boosting_result_list[3]
    x_list_name = boosting_result_list[4]
    y_name = x_list_name[0]
    causality_test_res = []
    print('------------Causalilty Test Criterias------------')

    # mse_y means the mse to predict y using all other varaibles except for the causing variable

    mse_y = mse_arr[len(mse_arr) - 2]
    #     print(mse_arr[len(mse_arr)-1])
    mse_all = mse_arr[len(mse_arr) - 1]

    print("mse before adding causing variable is ")
    print(mse_y)
    print("mse of all variables is")
    print(mse_all)
    print("\n!!!!!!!!!!!!!!!!!!!!!!!")
    print("change of mse -> np.log(mse_change)")
    mse_change = mse_y / mse_all
    #     mse_change = ((mse_y-mse_all)/(3-2))/(mse_all/(999-3))

    print(np.log(mse_change))
    print("!!!!!!!!!!!!!!!!!!!!!!!\n")

    print("~~~~~~~~~~~~~~~~~")
    print("the F-score is")
    f_score = ((mse_y - mse_all) / mse_all) * ((n - k * maxlag) / maxlag)
    print(n - k * maxlag)
    print(maxlag)
    print(k * maxlag)
    print(f_score)
    p_value = scipy.stats.f.sf(f_score, maxlag, n - k * maxlag)
    print("the p_value is")
    print(p_value)
    print("~~~~~~~~~~~~~~~~~")

    df['pred_y'] = df[name_list[0]]
    for key in range(1, len(name_list)):
        df['pred_y'] += df[name_list[key]]

    df['last_step'] = df['pred_y'] - df[name_list[len(name_list) - 1]]

    r2_y = r2_arr[len(r2_arr) - 2]
    #     print(mse_arr[len(mse_arr)-1])
    r2_all = r2_arr[len(r2_arr) - 1]

    print("r_square_last is")
    print(r2_y)
    print("r_square_final is ")
    print(r2_all)
    print("\n!!!!!!!!!!!!!!!!!!!!!!!")
    print("r-square change")
    r_square_change = abs(r2_all - r2_y) / r2_y
    print(r_square_change)
    print("!!!!!!!!!!!!!!!!!!!!!!!\n")

    if p_value < alpha:
        causality_test_res = [y_name, x_list_name[len(x_list_name) - 1], p_value, x_list_name]
        print(causality_test_res)

    return causality_test_res


def create_test_list(maxlag, input_list=x_list):
    algorithm_input_list = []
    y_x_list = []

    for idx in range(0, len(input_list)):
        # print("iteration {}".format)
        x = input_list[idx]
        # y is effect
        y = x
        tmp_list = input_list.remove(x)
        permutation_x_list = list(itertools.permutations(input_list))

        for permutation_x in permutation_x_list:
            permutation_x = list(permutation_x)
            permutation_x.insert(0, x)
            y_x_list.append([permutation_x, x, maxlag])

        input_list.insert(idx, x)
        # print(input_list)

    return y_x_list


test_list_name = create_test_list(maxlag, x_list)
print(test_list_name)

result = []

for iter_item in test_list_name:
    #     causality_test(boosting(iter_item[0], iter_item[1], iter_item[2]))
    result.append(causality_test(boosting(iter_item[0], iter_item[1], iter_item[2])))

for i in result[:]:
    if len(i) == 0:
        result.remove(i)

result_save = []
for item in result:
    # print(item[0])
    result_save.append(item)

with open("output_linear_{}.csv".format(data_file_name), "w", newline='') as f:
    for row in result_save:
        f.write("%s\n" % ','.join(str(col) for col in row))

print(datetime.now() - startTime)
