from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType,FloatType
from pyspark.sql import functions as F

def main():
    test_df, train_df, total_df = test_train_split()
    list_of_results = []
    list_of_results.append(linear_regression(test_df, train_df))
    list_of_results.append(generalized_linear_regression(test_df, train_df))
    list_of_results.append(random_forest_regression(test_df, train_df))
    list_of_results.append(decision_tree_regression(test_df, train_df))
    list_of_results.append(baseline_regression(test_df, train_df))
    print(list_of_results)
def test_train_split():
    spark_session = SparkSession.builder.master('local[*]').appName('session #?').getOrCreate()
    spark_context = spark_session._sc
    
    if False:
        chat_stat_without_month = "hdfs://localhost:8020/stats/chat_stats_ver2_"
        superchat_stat_without_month = "hdfs://localhost:8020/stats/superchat_stats_"
        
        chat_stat = spark_session.read.option("header",False) \
            .csv(chat_stat_without_month+str(2)+".csv") 
        chat_stat = chat_stat.withColumnRenamed("_c0","id") \
            .withColumnRenamed("_c1","period") \
            .withColumnRenamed("_c2","chats") \
            .withColumnRenamed("_c3","videoDays") \
            .withColumnRenamed("_c4","uniqueChatters") \
            .withColumnRenamed("_c5","totalBodyLength") 
        superchat_stat = spark_session.read.option("header",False) \
            .csv(superchat_stat_without_month+str(0)+".csv") 
        superchat_stat = superchat_stat.withColumnRenamed("_c0","id") \
            .withColumnRenamed("_c1","superchats") \
            .withColumnRenamed("_c2","avgSignificance")
        chat_stat.show()
        superchat_stat.show()
        print(superchat_stat.count())
        joint_stat = chat_stat.join(superchat_stat,chat_stat.id == superchat_stat.id, "right")
        joint_stat.show()
        print(joint_stat.count())

    import random
    # chat_list = [2,3,4,5,6,7,8,9,10,12,13,14,15]
    chat_list = [2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18]
    test_list  = random.sample(chat_list,5)
    train_list = list(set(chat_list) - set(test_list))
    print(test_list)
    print(train_list)
    
    chat_stat_without_month = "hdfs://localhost:8020/stats/chat_stats_ver2_"
    train_count = 0
    test_count = 0
    for i in chat_list:
        
        hadoop = spark_context._jvm.org.apache.hadoop
        fs = hadoop.fs.FileSystem
        conf = hadoop.conf.Configuration() 
        path_str = 'hdfs://localhost:8020/stats/combined_stats_'+str(i-2)+'.csv'
        path = hadoop.fs.Path(path_str)
        if fs.get(conf).exists(path):
            print('Combined stats exists!')
            joint_stat = spark_session.read.option("header",False) \
                .csv(path_str) 
            joint_stat = joint_stat.withColumnRenamed("_c0","period") \
                .withColumnRenamed("_c1","chats") \
                .withColumnRenamed("_c2","videoDays") \
                .withColumnRenamed("_c3","uniqueChatters") \
                .withColumnRenamed("_c4","totalBodyLength") \
                .withColumnRenamed("_c5","id") \
                .withColumnRenamed("_c6","superchats") \
                .withColumnRenamed("_c7","avgSignificance")
            col_type_map = {
                'chats': IntegerType(),
                'videoDays': IntegerType(),
                'uniqueChatters': IntegerType(),
                'totalBodyLength': IntegerType(),
                'superchats': FloatType(),
                'avgSignificance': FloatType()
            }
            joint_stat = joint_stat.select(
                [F.col(c).cast(col_type_map[c]) if c in col_type_map.keys() else c for c in joint_stat.columns]
            )
            joint_stat.show()
        else:
            chat_stat_without_month = "hdfs://localhost:8020/stats/chat_stats_ver2_"
            superchat_stat_without_month = "hdfs://localhost:8020/stats/superchat_stats_"
            
            chat_stat = spark_session.read.option("header",False) \
                .csv(chat_stat_without_month+str(i)+".csv") 
            chat_stat = chat_stat.withColumnRenamed("_c0","id") \
                .withColumnRenamed("_c1","period") \
                .withColumnRenamed("_c2","chats") \
                .withColumnRenamed("_c3","videoDays") \
                .withColumnRenamed("_c4","uniqueChatters") \
                .withColumnRenamed("_c5","totalBodyLength") 
            superchat_stat = spark_session.read.option("header",False) \
                .csv(superchat_stat_without_month+str(i-2)+".csv") 
            superchat_stat = superchat_stat.withColumnRenamed("_c0","id") \
                .withColumnRenamed("_c1","superchats") \
                .withColumnRenamed("_c2","avgSignificance")
            chat_stat.show()
            superchat_stat.show()
            print(superchat_stat.count())
            joint_stat = chat_stat.join(superchat_stat,chat_stat.id == superchat_stat.id, "inner").drop(chat_stat.id)
            col_type_map = {
                'chats': IntegerType(),
                'videoDays': IntegerType(),
                'uniqueChatters': IntegerType(),
                'totalBodyLength': IntegerType(),
                'superchats': FloatType(),
                'avgSignificance': FloatType()
            }
            joint_stat = joint_stat.select(
                [F.col(c).cast(col_type_map[c]) if c in col_type_map.keys() else c for c in joint_stat.columns]
            )
            joint_stat.show()
            print(joint_stat.count())
            joint_stat.coalesce(1).write.csv(path_str)
        
        if i in test_list:
            if test_count == 0:
                test_df = joint_stat
                test_count += 1
            else:
                test_df = test_df.union(joint_stat)
                test_count += 1
        elif i in train_list:
            if train_count == 0:
                train_df = joint_stat
                train_count += 1
            else:
                train_df = train_df.union(joint_stat)
                train_count += 1
                
    print(test_df.count())
    test_df.show()
    print(train_df.count())
    train_df.show()
    
    
    test_df = test_df.withColumn("log2_superchats", F.log2(F.col('superchats')))
    test_df = test_df.withColumn("log10_superchats", F.log10(F.col('superchats')))
    
    test_df.show()

    train_df = train_df.withColumn("log2_superchats", F.log2(F.col('superchats')))
    train_df = train_df.withColumn("log10_superchats", F.log10(F.col('superchats')))
    
    train_df.show()
    
    total_df = test_df.union(train_df)
    
    return test_df,train_df,total_df

def linear_regression(test_df, train_df,label_col = 'log10_superchats'):
    MAX_ITER = 20
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.mllib.evaluation import RegressionMetrics
    
    spark_session = SparkSession.builder.master('local[*]').appName('Linear Regression').getOrCreate()
    spark_context = spark_session._sc

    lr = LinearRegression(featuresCol = 'Independent Features', labelCol = label_col,maxIter=MAX_ITER, regParam=0.3, elasticNetParam=0.8)
    feature_assembler = VectorAssembler(inputCols = ["chats","videoDays","uniqueChatters","totalBodyLength"], outputCol = "Independent Features")
    test_output = feature_assembler.transform(test_df)
    train_output = feature_assembler.transform(train_df)

    train_output.select("Independent Features",label_col).show()
    # Fit the model
    lr_model = lr.fit(train_output)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(lr_model.coefficients))
    print("Intercept: %s" % str(lr_model.intercept))

    train_pred = lr_model.transform(train_output)
    test_pred = lr_model.transform(test_output)
    
    train_values_and_preds = train_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    train_values_and_preds = train_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(train_values_and_preds)
    # Squared Error
    print("Train MSE = %s" % metrics.meanSquaredError)
    print("Train RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Train MAE = %s" % metrics.meanAbsoluteError)
    
    result = {
        'model': 'linear regression',
        'label_col': label_col,
        'Train MSE' : metrics.meanSquaredError,
        'Train RMSE' : metrics.rootMeanSquaredError,
        'Train MAE' : metrics.meanAbsoluteError
    }
    
    test_values_and_preds = test_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    test_values_and_preds = test_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(test_values_and_preds)
    # Squared Error
    print("Test MSE = %s" % metrics.meanSquaredError)
    print("Test RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Test MAE = %s" % metrics.meanAbsoluteError)
    print("Test r-squared = %s" % metrics.r2)
    
    result['Test MSE'] = metrics.meanSquaredError
    result['Test RMSE'] = metrics.rootMeanSquaredError
    result['Test MAE'] = metrics.meanAbsoluteError
    
    lr_model.save(path='hdfs://localhost:8020/models/linear_regression_'+label_col)
    return result

def generalized_linear_regression(test_df, train_df,label_col = 'log10_superchats',family = 'tweedie'):
    FAMILY = family
    from pyspark.ml.regression import GeneralizedLinearRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.mllib.evaluation import RegressionMetrics
    spark_session = SparkSession.builder.master('local[*]').appName('Linear Regression').getOrCreate()
    spark_context = spark_session._sc

    glr = GeneralizedLinearRegression(featuresCol = 'Independent Features', labelCol = label_col,family=FAMILY, regParam=0.3)
    feature_assembler = VectorAssembler(inputCols = ["chats","videoDays","uniqueChatters","totalBodyLength"], outputCol = "Independent Features")
    test_output = feature_assembler.transform(test_df)
    train_output = feature_assembler.transform(train_df)

    train_output.select("Independent Features",label_col).show()
    # Fit the model
    glr_model = glr.fit(train_output)

    # Print the coefficients and intercept for linear regression
    print("Coefficients: %s" % str(glr_model.coefficients))
    print("Intercept: %s" % str(glr_model.intercept))

    train_pred = glr_model.transform(train_output)
    test_pred = glr_model.transform(test_output)
    
    train_values_and_preds = train_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    train_values_and_preds = train_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(train_values_and_preds)
    # Squared Error
    print("Train MSE = %s" % metrics.meanSquaredError)
    print("Train RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Train MAE = %s" % metrics.meanAbsoluteError)
    
    result = {
        'model': 'generalized linear regression',
        'label_col': label_col,
        'Train MSE' : metrics.meanSquaredError,
        'Train RMSE' : metrics.rootMeanSquaredError,
        'Train MAE' : metrics.meanAbsoluteError
    }
    
    test_values_and_preds = test_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    test_values_and_preds = test_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(test_values_and_preds)
    # Squared Error
    print("Test MSE = %s" % metrics.meanSquaredError)
    print("Test RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Test MAE = %s" % metrics.meanAbsoluteError)
    print("Test r-squared = %s" % metrics.r2)
    glr_model.save(path='hdfs://localhost:8020/models/generalized_linear_regression_'+label_col+'_'+family)
    
    result['Test MSE'] = metrics.meanSquaredError
    result['Test RMSE'] = metrics.rootMeanSquaredError
    result['Test MAE'] = metrics.meanAbsoluteError
    return result
def random_forest_regression(test_df, train_df,label_col = 'log10_superchats'):
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.ml.feature import VectorAssembler
    from pyspark.mllib.evaluation import RegressionMetrics
    spark_session = SparkSession.builder.master('local[*]').appName('Linear Regression').getOrCreate()
    spark_context = spark_session._sc

    rfr = RandomForestRegressor(featuresCol = 'Independent Features', labelCol = label_col,maxDepth = 6, maxBins = 64)
    feature_assembler = VectorAssembler(inputCols = ["chats","videoDays","uniqueChatters","totalBodyLength"], outputCol = "Independent Features")
    test_output = feature_assembler.transform(test_df)
    train_output = feature_assembler.transform(train_df)

    train_output.select("Independent Features",label_col).show()
    # Fit the model
    rfr_model = rfr.fit(train_output)

    train_pred = rfr_model.transform(train_output)
    test_pred = rfr_model.transform(test_output)
    
    train_values_and_preds = train_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    train_values_and_preds = train_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(train_values_and_preds)
    # Squared Error
    print("Train MSE = %s" % metrics.meanSquaredError)
    print("Train RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Train MAE = %s" % metrics.meanAbsoluteError)

    result = {
        'model': 'random forest regression',
        'label_col': label_col,
        'Train MSE' : metrics.meanSquaredError,
        'Train RMSE' : metrics.rootMeanSquaredError,
        'Train MAE' : metrics.meanAbsoluteError
    }
    
    test_values_and_preds = test_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    test_values_and_preds = test_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(test_values_and_preds)
    # Squared Error
    print("Test MSE = %s" % metrics.meanSquaredError)
    print("Test RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Test MAE = %s" % metrics.meanAbsoluteError)
    rfr_model.save(path='hdfs://localhost:8020/models/random_forest_regression_'+label_col)
    result['Test MSE'] = metrics.meanSquaredError
    result['Test RMSE'] = metrics.rootMeanSquaredError
    result['Test MAE'] = metrics.meanAbsoluteError
    return result

def decision_tree_regression(test_df, train_df,label_col = 'log10_superchats'):
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.feature import VectorAssembler
    from pyspark.mllib.evaluation import RegressionMetrics
    spark_session = SparkSession.builder.master('local[*]').appName('Linear Regression').getOrCreate()
    spark_context = spark_session._sc

    dtr = DecisionTreeRegressor(featuresCol = 'Independent Features', labelCol = label_col)
    feature_assembler = VectorAssembler(inputCols = ["chats","videoDays","uniqueChatters","totalBodyLength"], outputCol = "Independent Features")
    test_output = feature_assembler.transform(test_df)
    train_output = feature_assembler.transform(train_df)

    train_output.select("Independent Features",label_col).show()
    # Fit the model
    dtr_model = dtr.fit(train_output)

    train_pred = dtr_model.transform(train_output)
    test_pred = dtr_model.transform(test_output)
    
    train_values_and_preds = train_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    train_values_and_preds = train_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(train_values_and_preds)
    # Squared Error
    print("Train MSE = %s" % metrics.meanSquaredError)
    print("Train RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Train MAE = %s" % metrics.meanAbsoluteError)

    result = {
        'model': 'decision tree regression',
        'label_col': label_col,
        'Train MSE' : metrics.meanSquaredError,
        'Train RMSE' : metrics.rootMeanSquaredError,
        'Train MAE' : metrics.meanAbsoluteError
    }
    
    test_values_and_preds = test_pred.select([label_col, 'prediction'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    test_values_and_preds = test_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(test_values_and_preds)
    # Squared Error
    print("Test MSE = %s" % metrics.meanSquaredError)
    print("Test RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Test MAE = %s" % metrics.meanAbsoluteError)
    dtr_model.save(path='hdfs://localhost:8020/models/decision_tree_regression_'+label_col)
    result['Test MSE'] = metrics.meanSquaredError
    result['Test RMSE'] = metrics.rootMeanSquaredError
    result['Test MAE'] = metrics.meanAbsoluteError
    return result

def baseline_regression(test_df, train_df, label_col = 'log10_superchats'):
    import pyspark.sql.functions as F
    from pyspark.mllib.evaluation import RegressionMetrics
    
    test_mean = test_df.select(
        F.mean(F.col(label_col)).alias('mean')
    ).collect()
    test_mean = test_mean[0]['mean']
    train_mean = train_df.select(
        F.mean(F.col(label_col)).alias('mean')
    ).collect()
    train_mean = train_mean[0]['mean']
    test_df_2 = test_df.withColumn('mean',F.lit(test_mean))
    train_df_2 = train_df.withColumn('mean',F.lit(train_mean))
    
    train_values_and_preds = train_df_2.select([label_col, 'mean'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    train_values_and_preds = train_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(train_values_and_preds)
    # Squared Error
    print("Train MSE = %s" % metrics.meanSquaredError)
    print("Train RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Train MAE = %s" % metrics.meanAbsoluteError)
    
    result = {
        'model': 'baseline',
        'label_col': label_col,
        'Train MSE' : metrics.meanSquaredError,
        'Train RMSE' : metrics.rootMeanSquaredError,
        'Train MAE' : metrics.meanAbsoluteError
    }
    
    test_values_and_preds = test_df_2.select([label_col, 'mean'])
    # It needs to convert to RDD as the parameter of RegressionMetrics
    test_values_and_preds = test_values_and_preds.rdd.map(tuple)
    metrics = RegressionMetrics(test_values_and_preds)
    # Squared Error
    print("Test MSE = %s" % metrics.meanSquaredError)
    print("Test RMSE = %s" % metrics.rootMeanSquaredError)
    # Mean absolute error
    print("Test MAE = %s" % metrics.meanAbsoluteError)
    result['Test MSE'] = metrics.meanSquaredError
    result['Test RMSE'] = metrics.rootMeanSquaredError
    result['Test MAE'] = metrics.meanAbsoluteError
    return result
if __name__ == '__main__': 
    main()