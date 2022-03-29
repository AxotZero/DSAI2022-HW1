# DSAI-HW-2022

## 如何執行
```shell=
python3 app.py --training=data/training_data.csv --output=submission.csv
```


## Dataset
1.台灣電力公司_過去電力供需資訊2021  
2.台灣電力公司_本年度每日尖峰備轉容量率  

## Method
- 試過兩種方法
    分別為 ARIMA 和 GRU 做預測，最終由 GRU 些微的勝出，因此選擇使用 GRU  
    1. 使用ARIMA做預測  
        - 資料蒐集為台電2021年度加上今年度的備轉容量  
        - 經過不同時段預測，選擇距離3/29前120天當作訓練集，將數據decompose，呈現下圖走勢，分別為原數據、Trend、Seasonality 和  Stationary:  
            - ![](https://i.imgur.com/PHzMbGq.png)    
        - 針對ACF(Autocorrelation Function)和PACF(Partial Autocorrelation Function)圖可幫助我們判斷模型SARIMA(p, d, q)參數的選擇  
            - ![](https://i.imgur.com/KNyLsfo.png)    
        - 經過多次測量參數，最後參數挑選為order = (3,1,0)、seasonal_order=(1,0,1,12)，最終最好的 valid_rmse 為 **548**。  
        - 最後預測3/30~4/13的值，輸出在submission.csv  
    2. 使用GRU  
        1. 前處理  
            1. 從`台灣電力公司_本年度每日尖峰備轉容量率.csv` 和 `台灣電力公司_本年度每日尖峰備轉容量率.csv` 取出 '日期','備轉容量(MW)', '備轉容量率(%)' 這三個 feature 當作 training data  
            2. 新增`備轉容量(MW)` moving_average 的 features，windows=[3, 5, 7, 10, 15, 20]  
            3. min_max normalize  
            4. 在訓練時加 noise   
                ```python=
                x = x + (np.random.rand(*x.shape) - 0.5) * 0.1  
                ```
        2. 預測方法  
            1. 拿前30天，預測後面16天(因為我們沒有0329的資料，因此必須要預測16天)  
            2. 以 0313-0328的資料當作 validation 的 target，來作為選模型的依據，最終最好的 valid_rmse 為 **489.543**。  
            3. 預測 0329 - 0413 的結果，並捨棄掉 0329 的預測。  


