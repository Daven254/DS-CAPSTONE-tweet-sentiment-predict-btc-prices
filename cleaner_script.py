import pandas as pd
from ast import literal_eval
import os


import time
start_time = time.time()

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

date_1 = '2022-08-01'
date_2 = '2022-08-10'

Language_dict = {
    'arabic':'ar'
    ,'amharic':'am'
    ,'armenian':'hy'
    ,'bangla':'bn'
    ,'bulgarian':'bg'
    ,'burmese':'my'
    ,'central kurdish':'ckb'
    ,'chinese':'zh'
    ,'danish':'da'
    ,'divehi':'dv'
    ,'dutch':'nl'
    ,'english':'en'
    ,'estonian':'et'
    ,'finnish':'fi'
    ,'french':'fr'
    ,'georgian':'ka'
    ,'german':'de'
    ,'greek':'el'
    ,'gujarati':'gu'
    ,'haitian creol':'ht'
    ,'hebrew':'he'
    ,'hindi':'hi'
    ,'hungarian':'hu'
    ,'icelandic':'is'
    ,'indonesian':'id'
    ,'italian':'it'
    ,'japanese':'ja'
    ,'kannada':'kn'
    ,'khmer':'km'
    ,'korean':'ko'
    ,'lao':'lo'
    ,'latvian':'lv'
    ,'lithuanian':'lt'
    ,'malayalam':'ml'
    ,'marathi':'mr'
    ,'nepali':'ne'
    ,'norwegian':'no'
    ,'odia':'or'
    ,'pashto':'ps'
    ,'persian':'fa'
    ,'polish':'pl'
    ,'portugese':'pt'
    ,'punjabi':'pa'
    ,'romanian':'ro'
    ,'russian':'ru'
    ,'spanish':'es'
    ,'serbian':'sr'
    ,'sindhi':'sd'
    ,'sinhala':'si'
    ,'slovenian':'sl'
    ,'swedish':'sv'
    ,'tagalog':'tl'
    ,'tamil':'ta'
    ,'telugu':'te'
    ,'thai':'th'
    ,'tibetian':'bo'
    ,'turkish':'tr'
    ,'urdu':'ur'
    ,'uyghur':'ug'
    ,'vietnamese':'vi'
    ,'indonesian':'in'
    ,'czech':'cs'
    ,'welsh':'cy'
}


lang_dict = {value:key for key, value in Language_dict.items()}
del Language_dict
lang_dict['ca'] = 'english'
lang_dict['cz'] = 'czech'
lang_dict['iw'] = 'hebrew'
lang_dict['uk'] = 'english'
lang_dict['eu'] = 'english'

def convert_string_to_list(x):
    try:
        return literal_eval(str(x))   
    except Exception as e:
        print(e)
        return []


def read_crypto_df(name):
    df = pd.read_csv(name+'.csv',engine = 'python')
    if('user_id' in df.columns):
        df.drop('user_id',inplace = True, axis = 1)
    if ('Unnamed: 0' in df.columns):
        df.drop('Unnamed: 0',inplace = True, axis = 1)

    df.conversation_id = pd.to_numeric(df.conversation_id, errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    df['hashtags'] = df.hashtags.apply(lambda x: convert_string_to_list(x))
    
    if ('hashtags' in df.columns):
        df.drop('hashtags',inplace = True, axis = 1)
    
    df = df[df['language'] != 'und']
    df = df[df['language'] != 'qme']
    df = df[df['language'] != 'qht']
    df = df[df['language'] != 'qam']
    df = df[df['language'] != 'zxx']
    df = df[df['language'] != 'hy']
    df = df[df['language'] != 'te']
    df = df[df['language'] != 'lv']
    df = df[df['language'] != 'ta']
    df = df[df['language'] != 'am']
    df = df[df['language'] != 'my']
    df = df[df['language'] != 'ne']
    df = df[df['language'] != 'ta']
    df = df[df['language'] != 'he']
    df = df[df['language'] != 'pa']
    df = df[df['language'] != 'ug']
    df = df[df['language'] != 'iw']



    df = df.replace({'language':lang_dict})
    
    df = df.dropna()
    
    #df['hashtags'] = df.hashtags.apply(lambda x: literal_eval(str(x)))

    return df


def df_merge_tweets(name,df1,df2,date1,date2):
    df_merge = pd.concat([df1, df2],  axis=0, sort=True)
    df_merge.sort_values(by=['date'], inplace=True)
    df_merge.drop_duplicates(subset='conversation_id', keep="first")


    df_merge.reset_index(drop=True, inplace=True)


    df_merge = df_merge[(df_merge['date'] >= date1)]
    df_merge = df_merge[(df_merge['date'] < date2)]
    df_merge = df_merge.sort_values(by = 'date', ascending = False)
    
    df_merge.reset_index(drop=True, inplace=True)
    df_merge = df_merge.sort_values(by = 'date', ascending = True)
    
    df_merge = df_merge[df_merge.language != 'uyghur']
    df_merge = df_merge[df_merge.language != 'burmese'] 
    df_merge = df_merge[df_merge.language != 'telugu'] 
    df_merge = df_merge[df_merge.language != 'bangla'] 
    df_merge = df_merge[df_merge.language != 'marathi'] 
    df_merge = df_merge[df_merge.language != 'malayalam'] 
    df_merge = df_merge[df_merge.language != 'punjabi'] 
    df_merge = df_merge[df_merge.language != 'hebrew'] 
    df_merge = df_merge[df_merge.language != 'armenian'] 
    
    save_file = name+'_merged.csv'
    
    #df_btc_merged['pos'] = df_btc_merged.apply(lambda row: analyzer.polarity_scores(row.tweet)['pos'], axis = 1)
    #df_btc_merged['neg'] = df_btc_merged.apply(lambda row: analyzer.polarity_scores(row.tweet)['neg'], axis = 1)
    #df_btc_merged['neutral'] = df_btc_merged.apply(lambda row: analyzer.polarity_scores(row.tweet)['neu'], axis = 1)
    
    df_merge.to_csv(save_file)


    return df_merge



def output_csv(topic,dir_name):
    print(f'Starting Clean & Merging of Tweets and Price Data of {topic}...\nReading in First Tweets CSV\t\t\t',end = '')
    topic = topic

    #print('Reading in First Tweets CSV\t\t\t',end = '')
    read1_time = time.time()
    df_1 = read_crypto_df(topic)
    print('Read Complete:\t',"--- %s seconds ---" % (time.time() - read1_time),'\nReading in Second Tweets CSV\t\t\t',end = '')
    
    #print('Reading in Second Tweets CSV\t\t\t',end = '')
    read2_time = time.time()
    df_2 = read_crypto_df(topic+'_2')
    print('Read Complete:\t',"--- %s seconds ---" % (time.time() - read2_time),'\nMerging Tweets\t\t\t\t\t',end = '')

    #print('Merging Tweets\t\t\t',end = '')
    merge_time = time.time()
    df_merged = df_merge_tweets(topic,df_1,df_2,date_1,date_2)
    print('Merge Complete:\t', "--- %s seconds ---" % (time.time() - merge_time),'\nAnalyzing Sentiment\t\t\t\t',end = '')

    #print('Analyzing Sentiment\t\t',end = '')
    sentiment_time = time.time()
    df_merged['compound'] = df_merged.apply(lambda row: analyzer.polarity_scores(row.tweet)['compound'], axis = 1)
    print('Analysis Complete:', "--- %s seconds ---" % (time.time() - sentiment_time),f'\nReading in {dir_name} Price Tweets CSV\t\t\t',end = '')

    price_time = time.time()
    dir_name = dir_name
    col_name = ['date','open','high','low','close','close_time','volume','n_trades','taker_base','taker_quote','drop']

    
    price_df = pd.DataFrame()
    for x in os.listdir(dir_name):
        if('.csv' in x):
                df = pd.read_csv(dir_name+'/'+x, header=None, index_col = False)
                price_df = pd.concat([price_df,df], axis = 0)

    price_df = price_df.drop(columns = 11)
    price_df.columns = col_name
    price_df['date'] = pd.to_datetime(price_df['date'], unit='ms')
    print('Read Complete:\t', "--- %s seconds ---" % (time.time() - price_time),'\nMerging Price and Tweet CSVs\t\t\t',end = '')
    merge_time = time.time()
    price_df.sort_values(by = 'date')
    df_merged.sort_values(by = 'date')


    crypto = pd.merge_asof(df_merged,price_df.sort_values(by = 'date'),left_on = 'date',right_on = 'date',direction = 'nearest')
    drop_col = ['close_time','taker_base','taker_quote','drop']
    crypto.drop(drop_col, inplace = True, axis = 1)
    crypto['code'] = dir_name
    print('Merge Complete:\t', "--- %s seconds ---" % (time.time() - merge_time))

    crypto.set_index('date').to_csv('cleaned_'+topic+'_upto_'+date_2+'.csv',index = True)
    print('Outputting New CSV\t\t\t\t',end = '')

output_csv('bitcoin','btc')
print('Overall Cleaning Complete:', "--- %s seconds ---" % (time.time() - start_time))

                                                