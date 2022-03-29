import pandas as pd


COLUMNS_NAME = ['date', 'capacity', 'capacity_rate']
PAST_YEAR_PATH = '台灣電力公司_過去電力供需資訊2021.csv'
THIS_YEAR_PATH = '台灣電力公司_本年度每日尖峰備轉容量率.csv'

# load past year data
past_df = pd.read_csv(PAST_YEAR_PATH)
past_df = past_df[['日期','備轉容量(MW)', '備轉容量率(%)']]
past_df.columns = COLUMNS_NAME

# load this year data
now_df = pd.read_csv(THIS_YEAR_PATH)
now_df = now_df[['日期', '備轉容量(萬瓩)', '備轉容量率(%)']]
now_df.columns = COLUMNS_NAME
now_df = now_df[COLUMNS_NAME]
now_df['date'] = pd.to_datetime(now_df['date']).dt.strftime('%Y%m%d')
now_df['capacity'] = now_df['capacity'] * 10 # convert unit


# concat two files
df = pd.concat([past_df, now_df])
df.to_csv('training_data.csv', index=None)