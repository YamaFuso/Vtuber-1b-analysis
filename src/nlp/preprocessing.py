import string

from py4j.java_gateway import java_import
from pyspark import SparkContext
from pyspark.sql import SparkSession
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
import numpy as np
import pandas as pd
import os
from glob import glob
from deep_translator import DeeplTranslator, GoogleTranslator
from googletrans import Translator
import emoji
from pyspark.sql.types import *
import findspark

# Start SparkSession with Spark NLP
# start() functions has 3 parameters: gpu, m1, and memory
# sparknlp.start(gpu=True) will start the session with GPU support
# sparknlp.start(m1=True) will start the session with macOS M1 support
# sparknlp.start(memory="16G") to change the default driver memory in SparkSession
# spark = sparknlp.start(gpu=True)
# ResourceDownloader.showPublicPipelines(lang="en")

# os.environ['PYSPARK_SUBMIT_ARGS'] = 'spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.4'
# spark = SparkSession.builder \
#     .appName("Spark NLP")\
#     .master("local[*]")\
#     .config("spark.driver.memory","16G")\
#     .config("spark.driver.maxResultSize", "0") \
#     .config('spark.port.maxRetries', 100) \
#     .config("spark.kryoserializer.buffer.max", "2000M") \
#     .config("spark.jsl.settings.pretrained.cache_folder", "D:\ProgramData\cache_pretrained\pretrained") \
#     .config("spark.jsl.settings.storage.cluster_tmp_dir", "D:\ProgramData\cache_pretrained\storage") \
#     .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.4")\
#     .getOrCreate()

# # os.environ['PYSPARK_SUBMIT_ARGS'] = 'spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.4'
# SparkContext.setSystemProperty('spark.executor.memory', '8g')
# spark = sparknlp.start()

def read_file(df, path_nonflag, path_flag):
    df_nonflag, df_flag = [], []
    df_nonflag = pd.concat(
        [pd.read_parquet(x) for x in glob(path_nonflag)],
        ignore_index=True)
    df_flag = pd.concat(
        [pd.read_parquet(x) for x in glob(path_flag)],
        ignore_index=True)

    df = pd.concat([df_nonflag, df_flag])

    return df

def translate_mul_en_pipeline():
    documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")

    sentencerDL = SentenceDetectorDLModel() \
        .pretrained("sentence_detector_dl", "xx") \
        .setInputCols(["document"]) \
        .setOutputCol("sentences")

    marian = MarianTransformer.pretrained("opus_mt_it_en", "xx") \
        .setInputCols(["sentences"]) \
        .setOutputCol("translation")

    nlp_pipeline = Pipeline(stages=[
        documentAssembler,
        sentencerDL, marian
    ])

    return nlp_pipeline

def give_emoji_free_text(text):
    allchars = [str for str in text.decode('utf-8')]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.decode('utf-8').split() if not any(i in str for i in emoji_list)])
    return clean_text

def remove_punctuation(str):
    str_filtered = str.translate(str.maketrans('', '', string.punctuation))
    return str_filtered

if __name__ == "__main__":
    # path_nonflag = './archive/chats_nonflag_*.parquet'
    # path_flag = './archive/chats_flagged_*.parquet'
    # df = []
    # df = read_file(df, path_nonflag, path_flag)
    # df = df.drop_duplicates()
    # # flag: 1, nonflag: 0
    # df_labeled = df.copy()
    # df_labeled["label"] = np.where(df["label"].str.contains("nonflagged"), 0, 1)
    # df_labeled = df_labeled.sample(frac=1).reset_index(drop=True)
    # df_labeled.to_csv('chats.csv')
    # print(df_labeled)
    df_labeled = pd.read_csv('chats.csv')
    label_df = df_labeled[['label']]
    df_labeled_list = df_labeled[['body']].values.tolist()
    df_labeled_list = [''.join(map(str, x)) for x in df_labeled_list]
    # remove emoji
    df_labeled_list_free = [emoji.replace_emoji(x, replace='') for x in df_labeled_list]
    df_labeled_list_free = [x.strip() for x in df_labeled_list_free]
    df_labeled_list_free = [x for x in df_labeled_list_free if x.isalpha()]
    df_labeled_list_free = [remove_punctuation(x) for x in df_labeled_list_free]
    df_labeled_cleaned = pd.DataFrame({'body': df_labeled_list_free})
    df_labeled_cleaned = pd.concat([df_labeled_cleaned, label_df], axis=1)
    df_labeled_cleaned['body'].replace('', np.nan, inplace=True)
    df_labeled_cleaned.dropna(subset=['body'], inplace=True)
    # # df_labeled_list_free = [x.encode(encoding='ascii', errors='ignore').decode() for x in df_labeled_list_free]
    # df_labeled_cleaned = pd.DataFrame({'body': df_labeled_list_free})
    # df_labeled_cleaned = pd.concat([df_labeled_cleaned, label_df], axis=1)
    # df_labeled_cleaned.to_csv('chats_cleaned.csv', index=False)
    # # df_labeled_cleaned = pd.read_csv('chats_cleaned.csv')
    #
    # # df_sampled = df_labeled_cleaned.sample(n=50000)
    df_sampled_20000 = df_labeled_cleaned.sample(n=20000)
    df_sampled_50000 = df_labeled_cleaned.sample(n=50000)
    df_sampled_100000 = df_labeled_cleaned.sample(n=100000)
    df_sampled_20000.to_csv('chats_sample_20000.csv', index=False)
    df_sampled_50000.to_csv('chats_sample_50000.csv', index=False)
    df_sampled_100000.to_csv('chats_sample_100000.csv', index=False)
    # model = EasyNMT('opus-mt')
    # df_labeled_cleaned['translated'] = df_labeled_cleaned.apply(lambda x: model.translate(str(x['body']), source_lang='ja', target_lang='en'), axis=1)
    # translation
    # timeout = httpx.Timeout(10)
    # translator = Translator(timeout=timeout)
    # df_labeled_cleaned['translated'] = df_labeled_cleaned.apply(lambda x: translator.translate(x['body'], dest='en').text, axis=1)
    # # print(df_labeled_cleaned)
    # df_labeled_cleaned.to_csv('chats_translated.csv', index=False)
    # translations_list = translator.translate(df_labeled_list_free, dest='en')
    # df_translated = pd.DataFrame({'body': translations_list, 'label': label_list})
    # df_translated.to_csv('chats_translated.csv')
    # tranlation_pipeline = translate_mul_en_pipeline()
    # translation_model = tranlation_pipeline.fit(df_labeled[['body']])



