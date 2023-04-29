from flask import Flask, render_template, request, redirect, url_for
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import pickle
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import DatetimeTickFormatter
from bokeh.layouts import column
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from bs4 import BeautifulSoup
# from flask_socketio import SocketIO



app = Flask(__name__,static_folder='./static')
# app.secret_key = 'secret_key'
# socketio = SocketIO(app)
def build_model(inputs, output_size, neurons, activ_func="linear",
                dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model

def predict_1_day(LSTM_training_inputs,LSTM_training_outputs,LSTM_test_inputs,test_set):
    # random seed for reproducibility
    np.random.seed(202)
    # initialise model architecture
    bt_model = build_model(LSTM_training_inputs, output_size=1, neurons = 20)
    # train model on data
    # note: eth_history contains information on the training error per epoch
    bt_history = bt_model.fit(LSTM_training_inputs, 
                                LSTM_training_outputs, 
                                epochs=30, batch_size=1, verbose=2, shuffle=True)
    # Lưu trữ mô hình vào file
    bt_model.save('lstm_btc_model.h5')

    # Đóng gói mô hình và lưu trữ vào file
    with open('lstm_btc_model.pkl', 'wb') as f:
        pickle.dump(bt_model, f)
    #Predict
    result_1_day = bt_model(LSTM_test_inputs)
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(result_1_day)):
        denormalized_price = (result_1_day[i]+1) * test_set['bt_Close'].values[i] 
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_1 = np.array(predicted_prices_denormalized)

    return predicted_prices_1[-1:]
def predict_5_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len):
    # random seed for reproducibility
    np.random.seed(202)
    # we'll try to predict the closing price for the next 5 days 
    # change this value if you want to make longer/shorter prediction
    pred_range = 5
    # initialise model architecture
    bt_model_5_day = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)
    # model output is next 5 prices normalised to 10th previous closing price
    LSTM_training_outputs_5_day = []
    for i in range(window_len, len(training_set['bt_Close'])-pred_range):
        LSTM_training_outputs_5_day.append((training_set['bt_Close'][i:i+pred_range].values/
                                    training_set['bt_Close'].values[i-window_len])-1)
    LSTM_training_outputs_5_day = np.array(LSTM_training_outputs_5_day)
    # train model on data
    # note: eth_history contains information on the training error per epoch
    bt_history = bt_model_5_day.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs_5_day, 
                                epochs=40, batch_size=1, verbose=2, shuffle=True)
    # Lưu trữ mô hình vào file
    bt_model_5_day.save('lstm_btc_model_5_day.h5')

    # Đóng gói mô hình và lưu trữ vào file
    with open('lstm_btc_model_5_day.pkl', 'wb') as f:
        pickle.dump(bt_model_5_day, f)
    #Predict 5 day 
    result_5_day = bt_model_5_day(LSTM_test_inputs)
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(result_5_day)):
        denormalized_price = (result_5_day[i]+1) * test_set['bt_Close'].values[i] 
        # print(test_set['bt_Close'].values[i])
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_5 = np.array(predicted_prices_denormalized)

    return predicted_prices_5[-1:]
def predict_10_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len):
    # random seed for reproducibility
    np.random.seed(202)
    # we'll try to predict the closing price for the next 5 days 
    # change this value if you want to make longer/shorter prediction
    pred_range = 10
    # initialise model architecture
    bt_model_10_day = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)
    # model output is next 5 prices normalised to 10th previous closing price
    LSTM_training_outputs_10_day = []
    for i in range(window_len, len(training_set['bt_Close'])-pred_range):
        LSTM_training_outputs_10_day.append((training_set['bt_Close'][i:i+pred_range].values/
                                    training_set['bt_Close'].values[i-window_len])-1)
    LSTM_training_outputs_10_day = np.array(LSTM_training_outputs_10_day)
    # train model on data
    # note: eth_history contains information on the training error per epoch
    bt_history = bt_model_10_day.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs_10_day, 
                                epochs=40, batch_size=1, verbose=2, shuffle=True)
    # Lưu trữ mô hình vào file
    bt_model_10_day.save('lstm_btc_model_10_day.h5')

    # Đóng gói mô hình và lưu trữ vào file
    with open('lstm_btc_model_10_day.pkl', 'wb') as f:
        pickle.dump(bt_model_10_day, f)
    #Predict 10 day 
    result_10_day = bt_model_10_day(LSTM_test_inputs)
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(result_10_day)):
        denormalized_price = (result_10_day[i]+1) * test_set['bt_Close'].values[i] 
        # print(test_set['bt_Close'].values[i])
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_10 = np.array(predicted_prices_denormalized)

    return predicted_prices_10[-1:]
def predict_20_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len):
    # random seed for reproducibility
    np.random.seed(202)
    # we'll try to predict the closing price for the next 5 days 
    # change this value if you want to make longer/shorter prediction
    pred_range = 20
    # initialise model architecture
    bt_model_20_day = build_model(LSTM_training_inputs, output_size=pred_range, neurons = 20)
    # model output is next 5 prices normalised to 10th previous closing price
    LSTM_training_outputs_20_day = []
    for i in range(window_len, len(training_set['bt_Close'])-pred_range):
        LSTM_training_outputs_20_day.append((training_set['bt_Close'][i:i+pred_range].values/
                                    training_set['bt_Close'].values[i-window_len])-1)
    LSTM_training_outputs_20_day = np.array(LSTM_training_outputs_20_day)
    # train model on data
    # note: eth_history contains information on the training error per epoch
    bt_history = bt_model_20_day.fit(LSTM_training_inputs[:-pred_range], LSTM_training_outputs_20_day, 
                                epochs=40, batch_size=1, verbose=2, shuffle=True)
    # Lưu trữ mô hình vào file
    bt_model_20_day.save('lstm_btc_model_20_day.h5')

    # Đóng gói mô hình và lưu trữ vào file
    with open('lstm_btc_model_20_day.pkl', 'wb') as f:
        pickle.dump(bt_model_20_day, f)
    #Predict 20 day 
    result_20_day = bt_model_20_day(LSTM_test_inputs)
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(result_20_day)):
        denormalized_price = (result_20_day[i]+1) * test_set['bt_Close'].values[i] 
        # print(test_set['bt_Close'].values[i])
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_20 = np.array(predicted_prices_denormalized)

    return predicted_prices_20[-1:]
def predict(data):
    #EDA
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.dropna()
    data.columns =[data.columns[0]]+['bt_'+i for i in data.columns[1:]]
    #Take data from 2018
    market_info = data
    market_info = market_info[market_info['Date']>='2018-01-01']
    #Create new columns
    for coins in ['bt_']: 
        kwargs = { coins+'day_diff': lambda x: (x[coins+'Close']-x[coins+'Open'])/x[coins+'Open']}
        market_info = market_info.assign(**kwargs)
    for coins in ['bt_']: 
        kwargs = { coins+'close_off_high': lambda x: 2*(x[coins+'High']- x[coins+'Close'])/(x[coins+'High']-x[coins+'Low'])-1,
            coins+'volatility': lambda x: (x[coins+'High']- x[coins+'Low'])/(x[coins+'Open'])}
        market_info = market_info.assign(**kwargs)
    #Create data to build model
    model_data = market_info[['Date']+[coin+metric for coin in ['bt_'] 
                                   for metric in ['Close','Volume','close_off_high','volatility','day_diff']]]
    # need to reverse the data frame so that subsequent rows represent later timepoints
    model_data = model_data.sort_values(by='Date')
    #Split train and test 
    train_size=int(len(model_data) *0.8)
    test_size = int(len(model_data) *0.2)
    training_set, test_set = model_data[:train_size], model_data[train_size:]
    # training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)
    #Create window slide and norm columns
    window_len = 50
    norm_cols = [coin+metric for coin in ['bt_'] for metric in ['Close','Volume']]
    
    LSTM_training_inputs = []
    for i in range(len(training_set)-window_len):
        temp_set = training_set[i:(i+window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        LSTM_training_inputs.append(temp_set)
    LSTM_training_outputs = (training_set['bt_Close'][window_len:].values/training_set['bt_Close'][:-window_len].values)-1
    LSTM_test_inputs = []
    for i in range(len(test_set)-window_len):
        temp_set = test_set[i:(i+window_len)].copy()
        for col in norm_cols:
            temp_set.loc[:, col] = temp_set[col]/temp_set[col].iloc[0] - 1
        LSTM_test_inputs.append(temp_set)
    LSTM_test_outputs = (test_set['bt_Close'][window_len:].values/test_set['bt_Close'][:-window_len].values)-1
    
    # I find it easier to work with numpy arrays rather than pandas dataframes especially as we now only have numerical data
    LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
    LSTM_training_inputs = np.array(LSTM_training_inputs)

    LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
    LSTM_test_inputs = np.array(LSTM_test_inputs)
    #Build model and predict
    
    # result = {"predict_1_day": predict_1_day(LSTM_training_inputs,LSTM_training_outputs,LSTM_test_inputs,test_set), "predict_5_day": predict_5_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len),
    #           "predict_10_day": predict_10_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len), "predict_20_day": predict_20_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len)}
    
    #Dự đoán 1 ngày
    # Tải mô hình đã được đóng gói
    with open(r'C:\Users\Admin\OneDrive\Máy tính\SP_23\Model_predict\Bitcoin_web\lstm_btc_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Dự đoán với tập dữ liệu mới
    predictions = model.predict(LSTM_test_inputs)
    
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(predictions)):
        denormalized_price = (predictions[i]+1) * test_set['bt_Close'].values[i] 
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_1 = np.array(predicted_prices_denormalized)
    
    #Dự đoán 5 ngày
    # Tải mô hình đã được đóng gói
    with open(r'C:\Users\Admin\OneDrive\Máy tính\SP_23\Model_predict\Bitcoin_web\lstm_btc_model_5_day.pkl', 'rb') as f:
        model_5_day = pickle.load(f)

    # Dự đoán với tập dữ liệu mới
    predictions_5_day = model_5_day.predict(LSTM_test_inputs)
    
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(predictions_5_day)):
        denormalized_price = (predictions_5_day[i]+1) * test_set['bt_Close'].values[i] 
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_5 = np.array(predicted_prices_denormalized)
    
    #Dự đoán 10 ngày
    # Tải mô hình đã được đóng gói
    with open(r'C:\Users\Admin\OneDrive\Máy tính\SP_23\Model_predict\Bitcoin_web\lstm_btc_model_10_day.pkl', 'rb') as f:
        model_10_day = pickle.load(f)

    # Dự đoán với tập dữ liệu mới
    predictions_10_day = model_10_day.predict(LSTM_test_inputs)
    
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(predictions_10_day)):
        denormalized_price = (predictions_10_day[i]+1) * test_set['bt_Close'].values[i] 
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_10 = np.array(predicted_prices_denormalized)
    
    #Dự đoán 20 ngày
    # Tải mô hình đã được đóng gói
    with open(r'C:\Users\Admin\OneDrive\Máy tính\SP_23\Model_predict\Bitcoin_web\lstm_btc_model_20_day.pkl', 'rb') as f:
        model_20_day = pickle.load(f)

    # Dự đoán với tập dữ liệu mới
    predictions_20_day = model_20_day.predict(LSTM_test_inputs)
    
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(predictions_20_day)):
        denormalized_price = (predictions_20_day[i]+1) * test_set['bt_Close'].values[i] 
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_20 = np.array(predicted_prices_denormalized)
    
    result = {"predict_1_day":predicted_prices_1[-1:], "predict_5_day": predicted_prices_5[-1:], "predict_10_day": predicted_prices_10[-1:], "predict_20_day": predicted_prices_20[-1:]}
    
    return result
    
def get_table():
    url = 'https://coinmarketcap.com/'
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    rows = table.find_all('tr')[1:11]  # lấy thông tin của 10 đồng coin đầu tiên
    coins = []
    for row in rows:
        coin = {}
        coin['rank'] = row.find_all('td')[1].get_text().strip()
        coin['name'] = row.find_all('td')[2].find('p').get_text().strip() + " - " + row.find_all('td')[2].find_all('p')[-1].get_text().strip()
        coin['price'] = row.find_all('td')[3].get_text().strip()
        coin['change_1h'] = row.find_all('td')[4].get_text().strip()
        coin['change_24h'] = row.find_all('td')[5].get_text().strip()
        coin['change_7day'] = row.find_all('td')[6].get_text().strip()
        coin['market_cap'] = row.find_all('td')[7].find_all('span')[-1].text
        coin['volume_24h'] = row.find_all('td')[8].find('p').get_text()
        coin['circulating_supply'] = row.find_all('td')[9].get_text().strip()
        coins.append(coin)
    
    # print(coins)
    # Tạo DataFrame từ danh sách coins
    df_coins = pd.DataFrame(coins)
    data_list = df_coins.to_dict('records')
    return data_list


def dashboard(bitcoin_name):
    # # some code to get data
    # # Thực hiện crawl dữ liệu
    data = yf.download(bitcoin_name, period='max')
    data = data[:-1]
    data = data.reset_index().rename(columns={'index': 'Date'})
    data = data.sort_values(by=['Date'], ascending=False)
    data['Date'] = pd.to_datetime(data['Date'])
    # print(data.info())
    # create a figure with Bokeh
    p = figure(title="Bitcoin Price History",height=450, sizing_mode='stretch_width')
    p.line(data['Date'], data['Close'], line_width=2)
    # Cấu hình dashboard
    # layout = column(p, sizing_mode='fixed', width=800, height=300)

    xformatter = DatetimeTickFormatter(days="%Y-%m-%d")
    p.xaxis.formatter = xformatter
    # # generate the HTML code for the plot
    plot_html = file_html(p, CDN)
   
    return plot_html

def descripton(bitcoin_name):
    name = bitcoin_name[:-4] 
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/info"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '99f6663d-cd5c-4a75-8e1e-e68080bf23a5'  # thay bằng API của cá nhân
    }
    parameters = {
    'symbol': name
    }
    response = requests.get(url, headers=headers, params=parameters)
    data = response.json()['data']
    coin_info = data[name]
    try:
        message_board = coin_info[0]["urls"]["message_board"][0]
    except:
        message_board = coin_info[0]["urls"]["message_board"]
        
    try:
        reddit = coin_info[0]["urls"]["reddit"][0]
    except:
        reddit = coin_info[0]["urls"]["reddit"]
    try:
        source_code = coin_info[0]["urls"]["source_code"][0]
    except:
        source_code = coin_info[0]["urls"]["source_code"]
    try:
        technical_doc = coin_info[0]["urls"]["technical_doc"][0]
    except:
        technical_doc = coin_info[0]["urls"]["technical_doc"]
    try:
        website = coin_info[0]["urls"]["website"][0]
    except:
        website = coin_info[0]["urls"]["website"]
    description = {
        "description": coin_info[0]["description"],
        "logo": coin_info[0]["logo"],
        "message_board": message_board, # có thể bỏ [0]
        "reddit": reddit,
        "source_code": source_code,
        "technical_doc": technical_doc,
        "website": website,
        
        }
    
    url_1 = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
    headers_1 = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': '99f6663d-cd5c-4a75-8e1e-e68080bf23a5'
    }
    parameters_1 = {
    'symbol': name
    }
    response_1 = requests.get(url_1, headers=headers_1, params=parameters_1)
    data_1 = response_1.json()["data"]
    coin_info_1 = data_1[name]
    description_1 = {
        "name": coin_info_1[0]["name"],
        "slug": coin_info_1[0]["slug"],
        "symbol": coin_info_1[0]["symbol"],
        "cmc_rank": coin_info_1[0]["cmc_rank"],
        "circulating_supply": coin_info_1[0]["circulating_supply"],
        "date_added": coin_info_1[0]["date_added"],
        "last_updated": coin_info_1[0]["last_updated"],
        "max_supply": coin_info_1[0]["max_supply"],
        "price": coin_info_1[0]["quote"]["USD"]["price"],
        "change_1h": coin_info_1[0]["quote"]["USD"]["percent_change_1h"],
        "change_24h": coin_info_1[0]["quote"]["USD"]["percent_change_24h"],
        "change_7d": coin_info_1[0]["quote"]["USD"]["percent_change_7d"],
        "change_30d": coin_info_1[0]["quote"]["USD"]["percent_change_30d"],
        "change_60d": coin_info_1[0]["quote"]["USD"]["percent_change_60d"],
        "change_90d": coin_info_1[0]["quote"]["USD"]["percent_change_90d"],
        "volume_24h": coin_info_1[0]["quote"]["USD"]["volume_24h"],
        "volume_change_24h": coin_info_1[0]["quote"]["USD"]["volume_change_24h"]
        
        
    }
    output = {"description": description, "description_1": description_1}
    return output

NEWS_API_KEY = 'abd4488d9b074cf2b9d6836808c714cd'
# COIN = 'bitcoin'

def get_news(COIN):
    url = f"https://newsapi.org/v2/everything?q={COIN}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data['articles']
    news_list = []
    for article in articles:
        title = article['title']
        description = article['description']
        url = article['url']
        img_url = article['urlToImage']
        publishedAt = article['publishedAt']
        news_list.append({'title': title,'description': description, 'url':url, 'img_url': img_url , 'publishedAt': publishedAt})
    return news_list[:20]

# Route để hiển thị trang web với form nhập tên bitcoin
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Nếu form đã được submit, thực hiện crawl dữ liệu và chuyển hướng đến trang kết quả
        bitcoin_name = request.form['bitcoin_name']
        return redirect(url_for('show_results', bitcoin_name=bitcoin_name))
    else:
        # Nếu truy cập trang lần đầu, hiển thị form nhập tên bitcoin
        return render_template('./index_1.html')

# Route để hiển thị kết quả tìm kiếm
@app.route('/results/<bitcoin_name>')
def show_results(bitcoin_name):
    # Thực hiện crawl dữ liệu
    data = yf.download(bitcoin_name, period='max')
    data = data[:-1]
    # Chuyển đổi định dạng index của dữ liệu từ datetime sang string
    # data.index = data.index.strftime('%Y-%m-%d')
    data = data.reset_index().rename(columns={'index': 'Date'})
    data = data.sort_values(by=['Date'], ascending=False)

    # Chuyển dữ liệu thành list để truyền vào template
    data_list = data.to_dict('records')
    dashboard(bitcoin_name)
    
    data_table = get_table()
    news= get_news(bitcoin_name)
    # Save df to session
    # session['df'] = data_list

    des = descripton(bitcoin_name)
    # print(des)
    result_predict = predict(data)
    
    # Lấy ngày hiện tại
    today = datetime.datetime.today()
    
    #Hiển thị giá của 5 ngày tiếp theo
    next_date_5 = []
    # Hiển thị date của 5 ngày tiếp theo
    for i in range(0, 5):
        next_day = today + datetime.timedelta(days=i)
        next_date_5.append(next_day.date())
    
    # Tạo dataframe với 2 cột là 'date' và 'price'
    df_5_day = pd.DataFrame({'date': next_date_5, 'price': result_predict['predict_5_day'][0]})
    df_5_day_html = df_5_day.to_dict('records')
    # In ra dataframe
    # print(df_5_day)

    # create a figure with Bokeh
    predict_5 = figure(title="Bitcoin Price Predict",height=350, sizing_mode='stretch_width')
    predict_5.line(df_5_day['date'], df_5_day['price'], line_width=2)

    xformatter = DatetimeTickFormatter(days="%Y-%m-%d")
    predict_5.xaxis.formatter = xformatter
    # # generate the HTML code for the plot
    plot_html_5 = file_html(predict_5, CDN)
    
    #Hiển thị giá của 10 ngày tiếp theo 
    next_date_10 = []
    # Hiển thị date của 10 ngày tiếp theo
    for i in range(0, 10):
        next_day = today + datetime.timedelta(days=i)
        next_date_10.append(next_day.date())
    
    # Tạo dataframe với 2 cột là 'date' và 'price'
    df_10_day = pd.DataFrame({'date': next_date_10, 'price': result_predict['predict_10_day'][0]})
    df_10_day_html = df_10_day.to_dict('records')

    # create a figure with Bokeh
    predict_10 = figure(title="Bitcoin Price Predict",height=350, sizing_mode='stretch_width')
    predict_10.line(df_10_day['date'], df_10_day['price'], line_width=2)

    xformatter = DatetimeTickFormatter(days="%Y-%m-%d")
    predict_10.xaxis.formatter = xformatter
    # # generate the HTML code for the plot
    plot_html_10 = file_html(predict_10, CDN)
    
    #Hiển thị giá của 20 ngày tiếp theo
    next_date_20 = []
    # Hiển thị date của 5 ngày tiếp theo
    for i in range(0, 20):
        next_day = today + datetime.timedelta(days=i)
        next_date_20.append(next_day.date())
    
    # Tạo dataframe với 2 cột là 'date' và 'price'
    df_20_day = pd.DataFrame({'date': next_date_20, 'price': result_predict['predict_20_day'][0]})
    df_20_day_html = df_20_day.to_dict('records')

    # create a figure with Bokeh
    predict_20 = figure(title="Bitcoin Price Predict",height=350, sizing_mode='stretch_width')
    predict_20.line(df_20_day['date'], df_20_day['price'], line_width=2)

    xformatter = DatetimeTickFormatter(days="%Y-%m-%d")
    predict_20.xaxis.formatter = xformatter
    # # generate the HTML code for the plot
    plot_html_20 = file_html(predict_20, CDN)
    # Render template hiển thị kết quả
    return render_template('./index.html', data1= data_list, data2= des ,data3 = result_predict, data4 = df_5_day_html, data5 = df_10_day_html, data6 = df_20_day_html, data7= data_table, data8= news, plot1 = dashboard(bitcoin_name), plot2 = plot_html_5, plot3 = plot_html_10, plot4 = plot_html_20)

# @app.route('/dashboard')


if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     from waitress import serve
#     serve(app, host='127.0.0.1', port=5000)


# @socketio.on('connect')
# def test_connect():
#     print('Client connected')

# if __name__ == '__main__':
#     socketio.run(app)
