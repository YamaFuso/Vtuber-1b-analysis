import os
import time
from pathlib import Path
import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import sum,avg,max,min,count
def main():
    print('Spark initialization'.ljust(100,'-'))
    spark_session = SparkSession.builder.appName('initialize').getOrCreate()
    spark_context = spark_session._sc
    
    hadoop = spark_context._jvm.org.apache.hadoop

    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration() 
    path = hadoop.fs.Path('hdfs://localhost:8020/vtuber/')

    chat_path_list = []
    superchat_path_list = []
    
    for f in fs.get(conf).listStatus(path):
        str_f = str(f.getPath())
        # print(str_f)
        if '/chats_' in str_f and (not 'flagged' in str_f):
            chat_path_list.append(str_f)
        elif '/superchats_' in str_f:
            superchat_path_list.append(str_f)
    # print(superchat_path_list)
    
    print('Read events file into dataframes'.ljust(100,'-'))
    ban = spark_session.read.parquet('hdfs://localhost:8020/vtuber/ban_events.parquet')
    deletion = spark_session.read.parquet('hdfs://localhost:8020/vtuber/deletion_events.parquet')
    ban.show()
    deletion.show()
    spark_session.stop()
    # return 0
    
    print('Read chat file into dataframes'.ljust(100,'-'))
    # chat_list = []
    # for each_chat_path in chat_path_list:
    #     chat_list.append(spark_session.read.parquet(each_chat_path))
    
    index = 0
    time_list = []
    hadoop = spark_context._jvm.org.apache.hadoop
    fs = hadoop.fs.FileSystem
    conf = hadoop.conf.Configuration() 
    path = hadoop.fs.Path('hdfs://localhost:8020/stats/chat_stats*')
    # if fs.get(conf).exists(path):
    #     print('exists!')
    #     import subprocess
    #     subprocess.run(['hdfs', 'dfs', '-rm', '-r', 'hdfs://localhost:8020/stats/chat_stats*'])
        
    for each_chat_path in chat_path_list:
        print(('Starting spark session #'+str(index)).ljust(100,'-'))
        spark_session = SparkSession.builder.master('local[*]').appName('session #' + str(index)).getOrCreate()
        spark_context = spark_session._sc
        
        hadoop = spark_context._jvm.org.apache.hadoop
        fs = hadoop.fs.FileSystem
        conf = hadoop.conf.Configuration() 
        path_str = 'hdfs://localhost:8020/stats/chat_stats_alt'+str(index)+'.csv'
        path = hadoop.fs.Path(path_str)
        if fs.get(conf).exists(path):
            print('exists!')
            import subprocess
            subprocess.run(['hdfs', 'dfs', '-rm', '-r', path_str])
            # index += 1
            # continue
        if index == 3:
            break
        if index == 11: #corrupt file
            index += 1
            continue
            
        
        start_time = time.time()
        
        each_chat = spark_session.read.parquet(each_chat_path)
        each_chat.persist(ps.StorageLevel.MEMORY_AND_DISK)
        each_chat.show()
        each_chat.createOrReplaceTempView("chat")
        # result = spark_session.sql('''
        #     select distinct authorChannelId as channelId, first(SUBSTRING(to_date(timestamp), 1,7)) AS period,
        #     count(bodylength) AS chats, count(distinct DATE(timestamp)) AS videoDays, 
        #     count(distinct channelId) AS uniqueChatters, SUM(bodylength) AS totalBodyLength
        #     from chat 
        #     group by authorChannelId''')
        # result = spark_session.sql('''select authorChannelId from chat''')
        result = spark_session.sql('''
            SELECT t1.*, t2.uniqueMembers
            FROM
            (select distinct authorChannelId as channelId, first(SUBSTRING(to_date(timestamp), 1,7)) AS period,
            count(bodylength) AS chats, count(distinct DATE(timestamp)) AS videoDays, 
            count(distinct channelId) AS uniqueChatters, SUM(bodylength) AS totalBodyLength
            from chat 
            group by authorChannelId) t1
            LEFT JOIN
            (select distinct authorChannelId as channelID, count(distinct channelId) as uniqueMembers
            from chat
            where isMember = 'true'
            group by authorChannelId) t2
            on (t1.channelId=t2.channelId)''')#.cache()
        each_chat.unpersist(blocking=True)
        # result = result.na.fill(value=0,subset=["uniqueMembers"])
        
        
        
        result.coalesce(1).write.csv(path_str)
        spark_session.stop()
        index += 1
        cur_time = time.time()-start_time
        print('This run used '+ str(cur_time) + ' seconds')
        time_list.append(cur_time)  
    print(time_list)


    
if __name__ == '__main__': 
    main()