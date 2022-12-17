import os
from pathlib import Path
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum,avg,max,min,count

def main():
    # chats_and_superchats()
    models()

def chats_and_superchats():
    print('Read file into dataframe'.ljust(100,'-'))
    spark = SparkSession.builder.appName('stats')\
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:3.0.1').getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    # t_path = Path(os.path.dirname(os.path.realpath(__file__)),"t.csv").as_posix()
    
    print('Read and parse channels'.ljust(100,'-'))
    channels = spark.read.option("header",True) \
        .option("inferSchema", "true") \
        .csv("hdfs://localhost:8020/statistic/channels.csv") 
    channels = channels.withColumn("subscriptionCount", 
                                channels["subscriptionCount"]
                                .cast('int')) 
    channels = channels.withColumn("videoCount", 
                                channels["videoCount"]
                                .cast('int'))
    channels.printSchema()
    
    print('Read and parse normal chats'.ljust(100,'-'))
    from pyspark.sql.types import IntegerType
    from pyspark.sql import functions as F
    chat = spark.read.option("header",True).csv("hdfs://localhost:8020/statistic/chat_stats.csv")
    col_type_map = {
        'chats': IntegerType(),
        'memberChats': IntegerType(),
        'uniqueChatters': IntegerType(),
        'uniqueMembers': IntegerType(),
        'bannedChatters': IntegerType(),
        'deletedChats': IntegerType()
    }
    chat = chat.select(
        [F.col(c).cast(col_type_map[c]) if c in col_type_map.keys() else c for c in chat.columns]
    )
    chat.printSchema()

    print('Read and parse superchats'.ljust(100,'-'))
    superchat = spark.read.option("header",True).csv("hdfs://localhost:8020/statistic/superchat_stats.csv")
    col_type_map = {
        'superChats': IntegerType(),
        'uniqueSuperChatters': IntegerType(),
        'totalSC': IntegerType(),
        'averageSC': IntegerType(),
        'totalMessageLength': IntegerType()
    }
    superchat = superchat.select(
        [F.col(c).cast(col_type_map[c]) if c in col_type_map.keys() else c for c in superchat.columns]
    )
    superchat.printSchema()
    
    print('Join channel info with chat/superchat info'.ljust(100,'-'))
    channels = channels.drop('photo')
    channels_superchat_merged = channels.join(superchat, channels.channelId == superchat.channelId).drop(channels.channelId)
    channels_superchat_merged.printSchema()
    print(channels_superchat_merged.count())
    channels_chat_merged = channels.join(chat, channels.channelId == chat.channelId).drop(channels.channelId)
    channels_chat_merged.printSchema()
    
    
    print('Grouping superchat info'.ljust(100,'-'))
    from pyspark.sql import functions as F
    superchat_merged = channels_superchat_merged.groupBy('channelId')\
        .agg(F.first('name'),F.first('englishName'),F.first('affiliation'),F.first('subscriptionCount'), \
            F.first('videoCount'),F.collect_list('period'),F.collect_list('superChats'),F.collect_list('uniqueSuperChatters'), \
            F.collect_list('totalSC'),F.collect_list('averageSC'), F.collect_list('totalMessageLength'),F.collect_list('averageMessageLength'), \
            F.collect_list('mostFrequentCurrency'),F.collect_list('mostFrequentColor')    )
    col_list = ['channelId','name','englishName','affiliation','subscriptionCount',
                'videoCount','period','superChats','uniqueSuperChatters','totalSC',
                'averageSC','totalMessageLength','averageMessageLength','mostFrequentCurrency','mostFrequentColor']
    superchat_merged = superchat_merged.toDF(*col_list)
    superchat_merged.printSchema()
    # superchat_merged.show()

    print('Grouping chat info'.ljust(100,'-'))
    chat_merged = channels_chat_merged.groupBy('channelId')\
        .agg(F.first('name'),F.first('englishName'),F.first('affiliation'),F.first('subscriptionCount'), \
            F.first('videoCount'),F.collect_list('period'),F.collect_list('chats'),F.collect_list('memberChats'), F.collect_list('uniqueChatters'),\
            F.collect_list('uniqueMembers'),F.collect_list('bannedChatters'), F.collect_list('deletedChats'))
    col_list = ['channelId','name','englishName','affiliation','subscriptionCount',
                'videoCount','period','chats','memberChats','uniqueChatters',
                'uniqueMembers','bannedChatters','deletedChats']
    chat_merged = chat_merged.toDF(*col_list)
    chat_merged.printSchema()

    print('mongoDB setup'.ljust(100,'-'))
    from pymongo import MongoClient
    mongo_url = "mongodb+srv://maomao:panda123@ds503-final-project.1zndc50.mongodb.net/"
    mongoClient = MongoClient(mongo_url)
    stats_db = mongoClient['stats']
    list_of_collections = stats_db.list_collection_names()
    print(list_of_collections)
    mongoClient.close()
    
    print('JSONize superchat info and write to mongoDB'.ljust(100,'-'))
    if not 'superchat' in list_of_collections:
        print('collection superchat does not exist, writing to mongoDB'.ljust(100,'-'))
        superchat_merged.write\
        .format('com.mongodb.spark.sql.DefaultSource')\
        .option( "uri", mongo_url + "stats.superchat") \
        .save()
    else:
        print('collection superchat exists, skipping'.ljust(100,'-'))

    print('JSONize chat info and write to mongoDB'.ljust(100,'-'))
    if not 'chat' in list_of_collections:
        print('collection chat does not exist, writing to mongoDB'.ljust(100,'-'))
        chat_merged.write\
        .format('com.mongodb.spark.sql.DefaultSource')\
        .option( "uri", mongo_url + "stats.chat") \
        .save()
    else:
        print('collection chat exists, skipping'.ljust(100,'-'))

def models():
    raw_results = [
        {
            'model': 'linear regression',
            'label_col': 'superchats',
            'Train MSE' : 89.33711780510939,
            'Train RMSE' : 9.451831452428117,
            'Train MAE' : 3.829985822180715,
            'Test MSE' : 73.4862199336399,
            'Test RMSE' : 8.572410392278236,
            'Test MAE' : 3.639659455612303
        },
        {
            'model': 'glm gaussian',
            'label_col': 'superchats',
            'Train MSE' : 84.75455910585592,
            'Train RMSE' : 9.20622393307136,
            'Train MAE' : 3.7455358218073402,
            'Test MSE' : 83.58588852800251,
            'Test RMSE' : 9.142531844516732,
            'Test MAE' : 3.663929474875954
        },
        {
            'model': 'glm poisson',
            'label_col': 'superchats',
            'Train MSE' : 94.52395263548478,
            'Train RMSE' : 9.722342960186335,
            'Train MAE' : 3.7648116749884433,
            'Test MSE' : 156.42820023104233,
            'Test RMSE' : 12.507125978059161,
            'Test MAE' : 3.755809615380451
        },
        {
            'model': 'glm gamma',
            'label_col': 'superchats',
            'Train MSE' : 119.75828059203229,
            'Train RMSE' : 10.943412657486343,
            'Train MAE' : 4.461221201150051,
            'Test MSE' : 114.37318590897405,
            'Test RMSE' : 10.694540004552513,
            'Test MAE' : 4.314553869889432
        },
        {
            'model': 'glm tweedie',
            'label_col': 'superchats',
            'Train MSE' : 80.79272983484903,
            'Train RMSE' : 8.988477614971794,
            'Train MAE' : 3.5860437818764064,
            'Test MSE' : 92.92179880813966,
            'Test RMSE' : 9.639595365373987,
            'Test MAE' : 3.7429184695165922
        },
        {
            'model': 'rfr',
            'label_col': 'superchats',
            'Train MSE' : 86.16728657057267,
            'Train RMSE' : 9.282633601008534,
            'Train MAE' : 3.7437254802292594,
            'Test MSE' : 80.08146872705086,
            'Test RMSE' : 8.948824991419313,
            'Test MAE' : 3.6345815799747188
        },
        {
            'model': 'dtr',
            'label_col': 'superchats',
            'Train MSE' : 87.38166476569263,
            'Train RMSE' : 9.347816042568052,
            'Train MAE' : 3.798309772955146,
            'Test MSE' : 79.21786593751659,
            'Test RMSE' : 8.900441895631733,
            'Test MAE' : 3.646763688528405
        },
        {
            'model': 'baseline',
            'label_col': 'superchats',
            'Train MSE' : 96.09557245699725,
            'Train RMSE' : 9.802834919399452,
            'Train MAE' : 4.282734095332036,
            'Test MSE' : 103.23702540788129,
            'Test RMSE' : 10.160562258452103,
            'Test MAE' : 4.537985259171806
        }
    ]
    log_results = [
        {
            'model': 'linear regression',
            'label_col': 'log_superchats',
            'Train MSE' : 1.6563771904769666,
            'Train RMSE' : 1.28700318199955,
            'Train MAE' : 1.0330599445467163,
            'Test MSE' : 1.6572228988042985,
            'Test RMSE' : 1.2873316972732003,
            'Test MAE' : 1.0356546483332265
        },
        {
            'model': 'glm gaussian',
            'label_col': 'log_superchats',
            'Train MSE' : 1.6133659299134455,
            'Train RMSE' : 1.2701834237280243,
            'Train MAE' : 1.015709973342759,
            'Test MSE' : 1.4841709667419294,
            'Test RMSE' : 1.218265556741193,
            'Test MAE' : 0.9824545043239645
        },
        {
            'model': 'glm poisson',
            'label_col': 'log_superchats',
            'Train MSE' : 1.6255647301098355,
            'Train RMSE' : 1.2749763645298824,
            'Train MAE' : 1.0015114701528076,
            'Test MSE' : 1.6515642251386007,
            'Test RMSE' : 1.285131987438878,
            'Test MAE' : 1.0142981304922687
        },
        {
            'model': 'glm tweedie',
            'label_col': 'log_superchats',
            'Train MSE' : 1.5483572524514164,
            'Train RMSE' : 1.2443300416093057,
            'Train MAE' : 0.9939467652415913,
            'Test MSE' : 1.6296135893695098,
            'Test RMSE' : 1.2765631944285054,
            'Test MAE' : 1.0128697086595315
        },
        {
            'model': 'rfr',
            'label_col': 'log_superchats',
            'Train MSE' : 1.5082872722546785,
            'Train RMSE' : 1.228123475980603,
            'Train MAE' : 0.9677654003662401,
            'Test MSE' : 1.5304161388281994,
            'Test RMSE' : 1.237099890400205,
            'Test MAE' : 0.9751077954058505
        },
        {
            'model': 'dtr',
            'label_col': 'log_superchats',
            'Train MSE' : 1.5223044157602048,
            'Train RMSE' : 1.2338170106463133,
            'Train MAE' : 0.9703814333907987,
            'Test MSE' : 1.5318457126127818,
            'Test RMSE' : 1.2376775479149575,
            'Test MAE' : 0.9781022563708319
        },
        {
            'model': 'baseline',
            'label_col': 'log_superchats',
            'Train MSE' : 1.9979643597088816,
            'Train RMSE' : 1.4134936716196793,
            'Train MAE' : 1.13302703116848,
            'Test MSE' : 1.9919772738150785,
            'Test RMSE' : 1.4113742500892803,
            'Test MAE' : 1.1314595054244085
        }
    ]
    # import json
    # raw_results = json.dumps(raw_results, indent=2)
    # log_results = json.dumps(log_results, indent=2)
    print('mongoDB setup'.ljust(100,'-'))
    from pymongo import MongoClient
    mongo_url = "mongodb+srv://maomao:panda123@ds503-final-project.1zndc50.mongodb.net/"
    mongoClient = MongoClient(mongo_url)
    stats_db = mongoClient['stats']
    list_of_collections = stats_db.list_collection_names()
    print(list_of_collections)
    
    
    print('JSONize model info and write to mongoDB'.ljust(100,'-'))
    if not 'model' in list_of_collections:
        print('collection model does not exist, writing to mongoDB'.ljust(100,'-'))
        import pymongo
        model_col = stats_db["model"]

        x1 = model_col.insert_many(raw_results)
        x2 = model_col.insert_many(log_results)

        #print list of the _id values of the inserted documents:
        print(x1.inserted_ids)
        print(x2.inserted_ids)
    else:
        print('collection model exists, skipping'.ljust(100,'-'))
        
    mongoClient.close()
if __name__ == '__main__': 
    main()