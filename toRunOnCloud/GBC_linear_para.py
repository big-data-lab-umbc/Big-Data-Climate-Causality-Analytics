from __future__ import print_function
import sys
import scipy
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lag, col
from pyspark.sql.window import Window
from pyspark.sql.functions import desc, row_number, monotonically_increasing_id
from statsmodels.compat.python import (range, lrange, string_types,
                                       StringIO, iteritems)
import scipy.linalg
import scipy.stats as stats
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import itertools
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import csv
from datetime import datetime

startTime = datetime.now()
print("starting time: ", startTime)

# spark = SparkSession \
#     .builder.master('yarn') \
#     .appName("GB_Tree_Spark") \
#     .getOrCreate()

spark = SparkSession \
    .builder \
    .appName("GB_Tree_Spark") \
    .getOrCreate()

spark.sparkContext.setLogLevel("FATAL")

maxlag = int(sys.argv[1])
data_file_name = sys.argv[2]
thread_pool_num = int(sys.argv[3])
alpha = float(sys.argv[4])

data_input = spark.read.csv(data_file_name, header=True, inferSchema=True).cache()
k = len(data_input.columns)
data_input = data_input.withColumn("id", monotonically_increasing_id())

# add time lag for all x_name columns
w = Window().orderBy(col("id"))
x_list = data_input.columns
x_list.remove("id")
for x_name_item in x_list:
    for i in range(1, maxlag + 1):
        data_input = data_input.withColumn("%s_t-%s" % (x_name_item, str(i)),
                                           lag(data_input[x_name_item], i, 0).over(w))

data_input.cache()
n = data_input.count()


def regression(x_name, y_name, maxlag, data=data_input):
    print("!!!!!!!!!start regression!!!!!!!!!")
    print(x_name)
    print(y_name)
    v = 0.1
    data.printSchema()
    # data.show(10)
    # print(data.count())
    dataFrame = data

    input_feature_name = []
    for lagnumber in range(1, maxlag + 1):
        newname = "{}_t-{}".format(x_name, lagnumber)
        input_feature_name.append(newname)

    print("input_feature_name are")
    print(input_feature_name)

    assembler_for_lag = VectorAssembler(
        inputCols=input_feature_name,
        outputCol="features")

    lr = LinearRegression(featuresCol='features', labelCol='{}'.format(y_name), maxIter=1000, fitIntercept=True)
    pipeline = Pipeline(stages=[assembler_for_lag, lr])
    model = pipeline.fit(dataFrame)
    predictions = model.transform(dataFrame)

    # now predictions is the new dataFrame instead of the original dataFrame
    predictions = predictions.withColumnRenamed("prediction", 'predicted_{}'.format(y_name))
    mse = model.stages[1].summary.meanSquaredError

    # mse = evaluator.evaluate(predictions)
    print("Mean Squared Error (MSE) on test data = %g" % mse)

    y_hat = predictions.select('predicted_{}'.format(y_name))
    y_hat = y_hat.withColumn("yid", monotonically_increasing_id())
    # print(y_hat.count())

    # compute residual value of y, y-y_hat, the residual value is the y in next round of loop
    if y_name == x_name:
        # learning rate is not in model 0
        # dataFrame = dataFrame.join(y_hat, col("id") == (col("yid")+maxlag))
        dataFrame = dataFrame.join(y_hat, col("id") == col("yid"))
        residual = dataFrame['{}'.format(y_name)] - dataFrame['predicted_{}'.format(y_name)]
        dataFrame = dataFrame.withColumn("{}res{}".format(y_name, x_name), residual)
        # dataFrame.show(5)
        dataFrame = dataFrame.drop("yid")
        return_col = dataFrame.select("{}res{}".format(y_name, x_name))
        print("still round 1")
        # print(dataFrame.count())
    else:
        # apply leraning rate
        dataFrame = dataFrame.join(y_hat, col("id") == col("yid"))
        # dataFrame = dataFrame['predicted_{}'.format(y_name)]
        dataFrame = dataFrame.withColumn('v_predicted_{}'.format(y_name), col('predicted_{}'.format(y_name)) * v)
        # dataFrame.show(5)
        residual = dataFrame['{}'.format(y_name)] - dataFrame['v_predicted_{}'.format(y_name)]
        dataFrame = dataFrame.withColumn("{}res{}".format(y_name, x_name), residual)
        # dataFrame.show(5)
        dataFrame = dataFrame.drop("yid")
        return_col = dataFrame.select("{}res{}".format(y_name, x_name))
        print("after round 1 ")
        # dataFrame.show(5)
        # print(dataFrame.count())

    # n = dataFrame.count()
    # print("n is ")
    # print(n)
    print("data for next step is ")

    return return_col, mse


def boosting(x_list_name, y_name, maxlag, data=data_input):
    # loop through each variable in the list
    mse_arr = []
    r2_arr = []

    predicted_name_list = []

    for pivot_x in range(0, len(x_list_name)):
        print("=========this is regression round {}=========".format(pivot_x + 1))

        # save return value of regression in res_list
        if pivot_x == 0:
            res_list = regression(x_list_name[pivot_x], y_name, maxlag)
        else:
            res_col = res_list[0]

            res_col = res_col.withColumn("rid", monotonically_increasing_id())

            data = data.join(res_col, (res_col.rid == data.id))

            data = data.drop('rid')
            res_list = regression(x_list_name[pivot_x], y_name, maxlag, data)

        # # save predicted column name as a list
        predicted_name_list.append('predicted_{}'.format(y_name))
        #
        # # build y_name such as x1resx1, which means x1 substacts x1_hat, res means residual
        y_name = str(y_name) + "res" + str(x_list_name[pivot_x])

        print("in boosting def")
        print("this is round {}".format(pivot_x + 1))
        print('and yname as {}'.format(y_name))
        # # example: [0.7614110755692759, 0.6019695603895466, 0.4941602516989991, 0.36284165024184334]
        mse_arr.append(res_list[1])
        # r2_arr.append(res_list[3])

    return mse_arr, predicted_name_list, r2_arr, maxlag, x_list_name


def causality_test(boosting_result_list):
    mse_arr = boosting_result_list[0]
    name_list = boosting_result_list[1]
    r2_arr = boosting_result_list[2]
    maxlag = boosting_result_list[3]
    x_list_name = boosting_result_list[4]
    # y_name = boosting_result_list[7]

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
            # permutation_x = list(permutation_x).insert(0, y)
            permutation_x = list(permutation_x)
            permutation_x.insert(0, x)
            # print(permutation_x)
            y_x_list.append([permutation_x, x, maxlag])

        input_list.insert(idx, x)

    return y_x_list


# print("result:")
# print(create_test_list(maxlag, x_list))

test_list_name = create_test_list(maxlag, x_list)
# print(test_list_name)

for iterItem in test_list_name:
    print("=====")
    print(iterItem)

# solution: multiprocessing pool
pool = ThreadPool(thread_pool_num)

###
result = pool.map(lambda iter_item: causality_test(boosting(iter_item[0], iter_item[1], iter_item[2])), test_list_name)
# result_nn = filter(None, result)

print(result)

with open("out_linear_para.csv", "w", newline='') as f:
    for row in result:
        f.write("%s\n" % ','.join(str(col) for col in row))

with open('out_linear_para.csv') as finalInput, open('output_linear_para_{}.csv'.format(data_file_name), 'w',
                                                     newline='') as output:
    non_blank = (line for line in finalInput if line.strip())
    output.writelines(non_blank)

print(datetime.now() - startTime)

pool.close()
pool.join()

spark.catalog.clearCache()

# print(datetime.now() - startTime)
