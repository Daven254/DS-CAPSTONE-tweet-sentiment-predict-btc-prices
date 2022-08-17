#twint -s ethereum --until "2022-08-02 09:00:00" -o eth_data_2.csv --csv    

import pandas as pd
import twint 
import sys
#import nest_asyncio
#nest_asyncio.apply()
import os

print(os.getcwd())


topic = sys.argv[1]
pull = sys.argv[2]

c = twint.Config()
c.Search = topic
#c.Limit = 200
c.Pandas = True
c.Since = '2022-08-10 00:00:00'
#c.Until = '2022-08-04'
c.Pandas_clean = True
#c.Store_csv = True
#c.Custom = ['conversation_id', 'date','tweet', 'language', 'hashtags', 'user_id','username', 'nlikes', 'nreplies', 'nretweets']
# change the name of the csv file
#c.Output = "data/btc_data_pull.csv"


#c.Until = '2022-08-08 12:00:00'
#c.Resume = 'resume.txt'

twint.run.Search(c)

df = twint.storage.panda.Tweets_df
df_col = ['conversation_id', 'date','tweet', 'language', 'hashtags', 'user_id','username', 'nlikes', 'nreplies', 'nretweets']   


#df_col = df.columns
#print(df[df_columns_interest].head(20))

df = df[df_col]
df.to_csv(topic+'_'+pull+'.csv')
print('done')


