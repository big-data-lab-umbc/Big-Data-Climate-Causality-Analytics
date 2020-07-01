from __future__ import print_function
import sys
import scipy
import pyspark

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
from pyspark.sql.types import *
from pyspark.sql import SQLContext


startTime = datetime.now()
print("starting time: ", startTime)

# spark = SparkSession \
#     .builder.master('yarn') \
#     .appName("GB_Tree_Spark") \
#     .getOrCreate()

spark = SparkSession \
    .builder\
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
        data_input = data_input.withColumn("%s_t-%s" % (x_name_item, str(i)), lag(data_input[x_name_item], i, 0).over(w))

data_input.cache()

n = data_input.count()
nv = x_list[len(x_list)-1]
print(nv)
# print(x_list)
# maxlag = 3
# spark.stop()

del x_list[-1]
print(x_list)


def regression(x_name, y_name, maxlag, data=data_input):
    print("!!!!!!!!!start regression!!!!!!!!!")
    print(x_name)
    print(y_name)
    v = 0.1
    data.printSchema()
    # data.show(10)
    # print(data.count())
    dataFrame = data
    # if x_name == y_name:
    #     dataFrame = data.select(x_name)
    # else:
    #     dataFrame = data.select(x_name, y_name)
    #
    # dataFrame = dataFrame.withColumn("id", monotonically_increasing_id())
    # # dataFrame = dataFrame.withColumn('{}'.format(y_name), data.select(y_name))
    # # add time lag for x_name columns
    # w = Window().orderBy(col("id"))
    # for i in range(1, maxlag + 1):
    #     dataFrame = dataFrame.withColumn("%s_t-%s" % (x_name, str(i)), lag(dataFrame[x_name], i, 0).over(w))

    # roll max_lag rows to get rid of the 0s
    # dataFrame = dataFrame.withColumn("rid", monotonically_increasing_id())
    # TODO: now the roll back is commented out, check if this is needed after other debugging is done
    # maybe not, because in the join process in boosting function, the 0, 1, 2 will be removed
    # dataFrame = dataFrame.filter(dataFrame.rid >= maxlag)
    # print("====added rid columns ====")

    # dataFrame.show(10)
    # print(dataFrame.count())
    # dataFrame = dataFrame.drop('id')
    # dataFrame = dataFrame.drop('rid')
    input_feature_name = []
    for lagnumber in range(1,maxlag+1):
        newname = "{}_t-{}".format(x_name, lagnumber)
        input_feature_name.append(newname)
    # input_feature_name = dataFrame.schema.names

    # input_feature_name.remove("rid")
    # input_feature_name.remove("id")
    # input_feature_name.remove(x_name)
    # if not x_name == y_name:
    #     input_feature_name.remove(y_name)

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
    # print("predictions dataframe is ")
    # predictions.select('predicted_{}'.format(y_name), '{}'.format(y_name), "features").show(5)

    evaluator = RegressionEvaluator(
        labelCol='{}'.format(y_name), predictionCol='predicted_{}'.format(y_name), metricName="mse")
    mse = evaluator.evaluate(predictions)
    print("Mean Squared Error (MSE) on test data = %g" % mse)
    # featureImportances = model.stages[1].featureImportances
    # print("Feature Importance")
    # print(featureImportances)
    # spark.stop()

    # data = data.withColumn("rid", monotonically_increasing_id())
    # data = data.filter(data.rid >= maxlag)
    # print("====added rid columns ====")
    # #
    # print("!!!!!!!!!")
    # print(dataFrame.count())  # 996
    # print(predictions.count())  # 996
    # print("!!!!!!!!!")

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

    # k = len(data_ori.columns)

    # x_list = dataFrame.select(x_list_name)


    # x_list.show(5)

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
            # TODO: seems this join has some problem, in round 3, the dataframe size suddenly reduce to 4
            # TODO: check the id with the dataframe and the residual column that pass back
            # SOLVED: monotonically_increasing_id() is guaranteed to be increasing id from 0,1,2,3,4, ....
            # SOLVED: so I used row_number instead of that to ensure this step of join works
            # SOLVED: but not sure why it works in our regression function, if something gets wrong, will also apply this

            # join the residual value to the dataframe in next round
            # print(res_list[0])
            res_col = res_list[0]

            # df_update = res_list[1]

            # print("n is ")
            # print(n)

            res_col = res_col.withColumn("rid", monotonically_increasing_id())
            # res_col = res_col.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)
            # print("res_col count is ")
            # print(res_col.count())
            # res_col.show(10)

            # df_update = df_update.withColumn("id", monotonically_increasing_id())
            # df_update = df_update.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)
            # print("df update count is ")
            # print(df_update.count())
            # df_update.show(10)
            data = data.join(res_col, (res_col.rid == data.id))
            # data = res_col.join(df_update, "id")

            # outer join shows the super weired id numbers like 17179869184
            # joined_df = df_as1.join(df_as2, col("df_as1.name") == col("df_as2.name"), 'outer')

            # # data.show(10)
            # data = data.drop('id')
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

        if pivot_x == len(x_list_name)-1:
            print("I am saving the last record in g boosting ")
            return_result_col = res_col
            # return_result_col.show(5)
            # spark.stop()

    return mse_arr, predicted_name_list, r2_arr, maxlag, x_list_name, return_result_col


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
    # print("change of mse (ratio)")
    #     mse_change = mse_y/mse_all
    # mse_change = ((mse_y-mse_all)/(3-2))/(mse_all/(999-3))

    # print(np.log(mse_change))
    # print("!!!!!!!!!!!!!!!!!!!!!!!\n")

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

    return causality_test_res, boosting_result_list[5], mse_all
    # print(mse_all)
    # boosting_result_list[7].show(5)

# boosting(["x1", "x2", "x3", "x4"], "x1", 3)
# print(causality_test(boosting(["x1", "x2", "x3", "x4"], "x1", 3)))

# causality_test(boosting(["x3","x4","x2","x1"], "x3", 3))
# causality_test(boosting(["x3","x4","x1","x2"], "x3", 3))
# causality_test(boosting(["x3","x1","x2","x4"], "x3", 3))

def create_test_list(maxlag, input_list=x_list):
    algorithm_input_list = []
    y_x_list = []

    for idx in range(0, len(input_list)):
        # print("iteration {}".format)
        x = input_list[idx]
        # y is effect
        y = x
        tmp_list = input_list.remove(x)
        # print("===input _list =======")
        # print(input_list)
        # print("tmp_list")
        # print(tmp_list)
        # print("=====permutation======")
        permutation_x_list = list(itertools.permutations(input_list))
        # print("permutation_x_list")
        # print(permutation_x_list)

        for permutation_x in permutation_x_list:
            # permutation_x = list(permutation_x).insert(0, y)
            permutation_x = list(permutation_x)
            permutation_x.insert(0, x)
            # print(permutation_x)
            y_x_list.append([permutation_x, x, maxlag])
            # print(y_x_list)
            # y_x_list.append(permutation_x)
            # print(y_x_list)

        input_list.insert(idx, x)
        # print(input_list)

        # ('x1', 'x2', 'x3'), ('x1', 'x3', 'x2'), ('x2', 'x1', 'x3'), ('x2', 'x3', 'x1'), ('x3', 'x1', 'x2'), ('x3', 'x2', 'x1')
        # we want [4,1,2,3], 4, lag

        # for permutation_x in permutation_x_list:
        #     list(permutation_x).insert(0, y)
        #     y_x_list.append((list(permutation_x), y))
        #
        # algorithm_input_list.append(permutation_x_list)

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

### output result
result = pool.map(lambda iter_item: causality_test(boosting(iter_item[0], iter_item[1], iter_item[2])), test_list_name)
print("result is")
print(result)

for i in result[:]:
    if len(i[0]) == 0:
        result.remove(i)
print("result move i is")
print(result)

result_save = []
for item in result:
    print("item0 is ")
    print(item[0])
    result_save.append(item)


with open("out_exist_linear_para_incre.csv", "w", newline='') as f:
    for row in result_save:
        f.write("%s\n" % ','.join(str(col) for col in row[0]))

graphtime = datetime.now() - startTime
print(datetime.now() - startTime)

start_incremental_Time = datetime.now()
print("start incremental at ", start_incremental_Time)

def incremental_new_var_effect(new_x_name, list_name=test_list_name):
    y_x_list = []

    for iterItem in list_name:
        ### ["x3","x4","x2","x1"], "x3", 3]
        iterItem[0].insert(0, new_x_name)
        iterItem[1] = new_x_name
        y_x_list.append(iterItem)

    print(y_x_list)

    return y_x_list



# # incremental_gbt_new_var_causing('x4', 3)
incremental_gbt_new_var_effect_list = incremental_new_var_effect(nv)

incremental_new_var_effect_res = pool.map(lambda iter_item: causality_test(boosting(iter_item[0], iter_item[1], iter_item[2])),
                                          incremental_gbt_new_var_effect_list)


def incremental_gbt_new_var_as_causing_causality(new_x_name, maxlag, old_result, data=data_input, incre_k = k):

    ### every other variable causes x  (x,a)

    # old_list, old_mse, return_col
    # fit return_col to new variable
    # get mse, then run causality test
    print(old_result)
    causality_test_res = []
    old_pred = old_result[1]
    old_mse = old_result[2]

    y_name = old_pred.columns[0]
    print(y_name)

    new_x_name = nv

    # TODO: join the return pred column to data
    # old_pred = old_pred.withColumn('pid', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)
    old_pred = old_pred.withColumn("pid", monotonically_increasing_id())
    # data = data.withColumn('id', row_number().over(Window.orderBy(monotonically_increasing_id())) - 1)
    data = data.join(old_pred, (data.id == old_pred.pid))
    # data.show(10)
    # return_col, data, mse, n, featureImportances
    regression_res = regression(new_x_name, y_name, maxlag, data)

    ## causality test using  xxx
    # f_score = ((mse_y - mse_all) / mse_all) * ((n - k * maxlag) / maxlag)
    # old_mse → mse_y
    # regression_res[2] = mse_all
    # n → regression_res[3]  number of rows
    # k → len(x_list)+1  input lenth + 1, 1 is the new variable

    mse_y = old_mse
    mse_all = regression_res[1]
    incre_k = incre_k + 1
    print(incre_k)
    print("~~~~~~~~~~~~~~~~~")
    print("the F-score is")
    f_score = ((mse_y - mse_all) / mse_all) * ((n - incre_k * maxlag) / maxlag)
    print(n - incre_k * maxlag)
    print(maxlag)
    print(incre_k * maxlag)
    print(f_score)
    p_value = scipy.stats.f.sf(f_score, maxlag, n - incre_k * maxlag)
    print("the p_value is")
    print(p_value)
    print("~~~~~~~~~~~~~~~~~")

    # if p > threshold 0.05, save edge
    if p_value < alpha:
        causality_test_res = [y_name, new_x_name, p_value, old_result[0][3].insert(len(old_result[0][3]),new_x_name)]
        print(causality_test_res)

    return causality_test_res

new_variable = nv
incremental_gbt_new_var_causing_res = pool.map(lambda iter_item: incremental_gbt_new_var_as_causing_causality(new_variable, maxlag, iter_item), result)

print("================================================")
print("incremental results:")
print("as effect:")
print(incremental_new_var_effect_res)
print("as causing:")
print(incremental_gbt_new_var_causing_res)
print("================================================")


for incre_effect_item in incremental_new_var_effect_res[:]:
    if len(incre_effect_item[0]) == 0:
        incremental_new_var_effect_res.remove(incre_effect_item)

for incre_cause_item in incremental_gbt_new_var_causing_res[:]:
    if len(incre_cause_item) == 0:
        incremental_gbt_new_var_causing_res.remove(incre_cause_item)

incre_effect_item_save = []
for item in incremental_new_var_effect_res:
    incre_effect_item_save.append(item)

incre_causing_item_save = []
for item in incremental_gbt_new_var_causing_res:
    incre_causing_item_save.append(item)

with open("out_linear_para_incre_{}.csv".format(data_file_name), "w", newline='') as f:
    for row in incre_effect_item_save:
        f.write("%s\n" % ','.join(str(col) for col in row[0]))
    for row2 in incre_causing_item_save:
        f.write("%s\n" % ','.join(str(col2) for col2 in row2[0]))

with open('out_exist_linear_para_incre.csv') as finalInput, open('out_linear_para_incre_{}.csv'.format(data_file_name), 'a', newline='') as finalOutput:
    for rowFinal in finalInput:
        finalOutput.write(rowFinal)

print("total time")
print(datetime.now() - startTime)
print("graph time")
print(graphtime)
print("incre_time")
print(datetime.now() - start_incremental_Time)

pool.close()
pool.join()

spark.catalog.clearCache()
