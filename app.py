import pandas as pd
import argparse
from datetime import datetime
from tqdm import tqdm
# import matplotlib.pyplot as plt 
# from pmdarima import auto_arima
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')


def run_arima(data):
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima_model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    NUM_PRED_DATE = 15

    train = data[:-NUM_PRED_DATE]
    test = data[-NUM_PRED_DATE:]

    train.set_index('date',inplace=True)
    # test.set_index('date',inplace=True)

    predictions = []

    #forecast 
    for t in range(len(test)):
        someday = train[-1:].index.values[0]
        someday = pd.to_datetime(someday, format = '%Y-%m-%d %H:%M:%S')
        # calculating end date by adding 1 day
        Enddate = someday + timedelta(days=1)
        
        model = ARIMA(train, order=(2,1,3))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast()[0]
        
        predictions.append(yhat)
        train.loc[Enddate] = [round(yhat[0], 1)]

    rmse = sqrt(mean_squared_error(test.iloc[:, 1], predictions))
    print('Test RMSE: %.3f'%rmse)

    result_predict = test[-NUM_PRED_DATE:].copy()
    result_predict.columns = ['date', 'operating_reserve(MW)']
    result_predict.to_csv('submission.csv', index=None)


def run_gru(data_path, output_path):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import random
    # from ranger import Ranger
    from pdb import set_trace
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    
    class MyModel(nn.Module):
        def __init__(self, num_feat, num_day_pred=15):
            super().__init__()
            self.num_day_pred = num_day_pred
            self.num_feat = num_feat
            hidden_size = 32
            self.gru = nn.GRU(
                input_size=num_feat, 
                hidden_size=hidden_size,
                dropout=0.5,
                num_layers=3,
                batch_first=True
            )
            self.next_input_regressor = nn.Linear(hidden_size, self.num_feat)
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

        def forward(self, seq_data):
            # seq_data shape: (bs, num_day+num_day_pred, num_feat)
            ## with padding
            embs, _ = self.gru(seq_data)
            embs = embs[:, -self.num_day_pred:]
            preds = self.regressor(embs).squeeze()

            # w/o padding
            # embs, h_t = self.gru(seq_data)
            # # preds = [embs[:, i].unsqueeze(1) for i in range(embs.size()[1])]
            # preds = [embs[:, -1].unsqueeze(1)]
            # for i in range(self.num_day_pred-1):
            #     embs = embs[:, -1].unsqueeze(1)
            #     embs = self.next_input_regressor(embs)
            #     embs, h_t = self.gru(embs, h_t)
            #     preds.append(embs[:, -1].unsqueeze(1))
            # preds = torch.cat(preds, dim=1)
            # preds = self.regressor(preds).squeeze()
            return preds
    
    def padding(seq_data, num_day_pred=15):
        padding_value = np.ones((num_day_pred, seq_data.shape[1])) * -1
        return np.concatenate((seq_data, padding_value), axis=0)

    def normalize(data, _min, _max):
        return (data - _min)/(_max - _min)
    
    def denormalize(data, _min, _max):
        return data * (_max - _min) + _min

    class MyDataset(Dataset):
        def __init__(self, data, num_day_input=15, num_day_pred=15, training=True):
            self.num_day_pred = num_day_pred
            self.num_day_input = num_day_input
            self.data = data
            self.training = training # train, valid, test
        
        def __len__(self):
            return len(self.data) - self.num_day_pred - self.num_day_input + 1

        def __getitem__(self, x_start_idx):
            x_end_idx = x_start_idx + self.num_day_input
            x = self.data[x_start_idx : x_end_idx]
            if self.training:
                x = x + (np.random.rand(*x.shape) - 0.5) * 0.1

            x = padding(x, self.num_day_pred)
            # y = self.data[x_start_idx + 1 : x_end_idx+self.num_day_pred][:, 0]
            y = self.data[x_end_idx : x_end_idx+self.num_day_pred][:, 0]
            return x, y

    
    EPOCH = 300
    LR = 1e-3
    BATCH_SIZE = 128
    NUM_DAY_INPUT = 30
    NUM_DAY_PRED = 16
    SAVE_MODEL_PATH = 'model.pt'

    # add moving average
    raw = pd.read_csv(data_path)
    data = raw.copy()
    for window in [3, 5, 7, 10, 15, 20]:
        data[f'ma{window}'] = data['capacity'].rolling(window).mean()
    data = data.iloc[:, 1:].dropna().values

    # split data
    train_data = data[: -NUM_DAY_PRED]
    valid_data = data[-(NUM_DAY_PRED+NUM_DAY_INPUT):]
    test_data = data[-NUM_DAY_INPUT:]

    # normalize
    _min, _max = np.min(train_data, axis=0), np.max(train_data, axis=0)
    train_data = normalize(train_data, _min, _max)
    valid_data = normalize(valid_data, _min, _max)
    test_data = normalize(test_data, _min, _max)

    # dataloader
    train_dataloader = DataLoader(
        MyDataset(train_data, NUM_DAY_INPUT, NUM_DAY_PRED, training=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    valid_dataloader = DataLoader(
        MyDataset(valid_data, NUM_DAY_INPUT, NUM_DAY_PRED, training=False),
        batch_size=1
    )
    
    def rmse(yhat, y):
        return torch.sqrt(torch.mean((yhat-y)**2))

    # training
    model = MyModel(num_feat=data.shape[-1], num_day_pred=NUM_DAY_PRED)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # optimizer = Ranger(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    # criterion = rmse

    best_score = float("inf")
    for epoch in range(EPOCH):
        # train
        model.train()
        for x, y in train_dataloader:
            # pred
            pred = model(x.float())

            # compute loss
            # set_trace()
            loss = criterion(pred, y.float())

            # back ward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # valid
        model.eval()
        for x, y in valid_dataloader:
            # pred
            with torch.no_grad():
                pred = model(x.float())

            # denormalize
            pred = pred.detach()
            y = y.float().detach()
            pred = denormalize(pred, _min[0], _max[0])
            y = denormalize(y, _min[0], _max[0])
            
            # compute loss
            loss = rmse(pred, y).item()
            print(f'Valid Epoch: {epoch+1}, rmse: {loss :.1f}')

            if loss < best_score:
                best_score = loss
                torch.save(model.state_dict(), SAVE_MODEL_PATH)

    print(f'best_score: {best_score}')

    print(f' === Run Test === ')
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    pred = model(torch.tensor(test_data).float().unsqueeze(0))
    pred = denormalize(pred, _min[0], _max[0]).squeeze().detach().numpy()
    sub_df = pd.DataFrame(
        {
            'date': pd.date_range(start="20220330", end="20220413"),
            'operating_reserve(MW)': pred[-15:]
        }
    )
    sub_df.to_csv(output_path, index=None)


# You can write code above the if-main block.
if __name__ == '__main__':

    # # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='data/training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    parser.add_argument('--model', 
                        default='gru',
                        choices=['arima', 'gru'])
    args = parser.parse_args()


    
    # The following part is an example.
    # You can modify it at will.
    
    eval(f'run_{args.model}')(args.training, args.output)

