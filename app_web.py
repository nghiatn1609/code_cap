# # -*- coding: utf-8 -*-

# import numpy as np
# import pickle
# from flask import Flask, request, render_template

# # Load ML model
# # model = pickle.load(open('model_pre.pkl', 'rb')) 

# # Create application
# app = Flask(__name__,template_folder='templates')

# # Bind home function to URL
# @app.route('/')
# def home():
#     return render_template('./index.html')

# # Bind predict function to URL
# @app.route('/predict', methods =['POST'])

# def predict():
    
#     # Put all form entries values in a list 
#     features = [float(i) for i in request.form.values()]
#     # Convert features to array
#     array_features = [np.array(features)]
#     # Predict features
#     prediction = model.predict(array_features)
    
#     output = prediction
    
#     # Check the output values and retrive the result with html tag based on the value
#     return render_template('index.html', result = output)

# if __name__ == '__main__':
# #Run the application
#     app.run()


# thử lần 1
# from flask import Flask, render_template
# import yfinance as yf
# import datetime


# app = Flask(__name__)

# @app.route('/')
# def index():
#     start_date = datetime.datetime(2010, 1, 1)
#     end_date = datetime.datetime.now()

#     data = yf.download('BTC-USD', start=start_date, end=end_date, group_by='day')
#     # print(data)
#     return render_template('./index_data.html', data=data)

# if __name__ == '__main__':
#     app.run()

# thử lần 2
# from flask import Flask, render_template
# import yfinance as yf

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('./index_1.html')

# @app.route('/table', methods=['POST'])
# def table():
#     # Crawl dữ liệu từ finance yahoo thông qua yfinance
#     ticker = yf.Ticker('BTC-USD')
#     df = ticker.history(period='max')
#     df = df.reset_index().rename(columns={'index': 'Date'})
#     # df['Date'] = df['Date'].strftime("%d-%m-%Y")
#     # Convert dataframe thành list các dictionary để dễ truy xuất trong template
#     data = df.to_dict('records')

#     # Render template và truyền dữ liệu vào
#     return render_template('./index_data.html', data=data)

# if __name__ == '__main__':
#     app.run(debug=True)


# thử lần 3 
from flask import Flask, render_template, request, redirect, url_for , jsonify
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
# import the relevant Keras modules
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model



app = Flask(__name__,static_folder='./static')
app.secret_key = 'secret_key'

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
    # # Lưu trữ mô hình vào file
    # bt_model.save('lstm_btc_model.h5')

    # # Đóng gói mô hình và lưu trữ vào file
    # with open('lstm_btc_model.pkl', 'wb') as f:
    #     pickle.dump(bt_model, f)
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
    # # Lưu trữ mô hình vào file
    # bt_model_5_day.save('lstm_btc_model_5_day.h5')

    # # Đóng gói mô hình và lưu trữ vào file
    # with open('lstm_btc_model_5_day.pkl', 'wb') as f:
    #     pickle.dump(bt_model_5_day, f)
    #Predict 5 day 
    result_5_day = bt_model_5_day(LSTM_test_inputs[:-pred_range])
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(result_5_day)):
        denormalized_price = (result_5_day[i]+1) * test_set['bt_Close'].values[i] 
        # print(test_set['bt_Close'].values[i])
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_5 = np.array(predicted_prices_denormalized)

    return predicted_prices_5[-1:]
def predict(data):
    #EDA
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.dropna()
    data.columns =[data.columns[0]]+['bt_'+i for i in data.columns[1:]]
    #Take data from 2017
    market_info = data
    market_info = market_info[market_info['Date']>='2017-01-01']
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
    split_date = '2022-01-01'
    training_set, test_set = model_data[model_data['Date']<split_date], model_data[model_data['Date']>=split_date]
    training_set = training_set.drop('Date', 1)
    test_set = test_set.drop('Date', 1)
    #Create window slide and norm columns
    window_len = 20
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
    # predict_1_day = []
    # predict_5_day = []
    # predict_1_day= predict_1_day(LSTM_training_inputs,LSTM_training_outputs,LSTM_test_inputs,test_set)
    # predict_5_day= predict_5_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len)
    
    # result = {"predict_1_day": predict_1_day(LSTM_training_inputs,LSTM_training_outputs,LSTM_test_inputs,test_set), "predict_5_day": predict_5_day(LSTM_training_inputs,LSTM_test_inputs,training_set,test_set,window_len)}
    
    # Tải mô hình đã được đóng gói
    with open(r'C:/Users/Admin/OneDrive/Máy tính/SP_23/Model_predict/Bitcoin_web/lstm_btc_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Dự đoán với tập dữ liệu mới
    predictions = model.predict(LSTM_test_inputs)
    
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(predictions)):
        denormalized_price = (predictions[i]+1) * test_set['bt_Close'].values[i] 
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_1 = np.array(predicted_prices_denormalized)
    
    # Tải mô hình đã được đóng gói
    with open(r'C:/Users/Admin/OneDrive/Máy tính/SP_23/Model_predict/Bitcoin_web/lstm_btc_model_5_day.pkl', 'rb') as f:
        model_5_day = pickle.load(f)

    # Dự đoán với tập dữ liệu mới
    predictions_5_day = model_5_day.predict(LSTM_test_inputs[:-5])
    
    # đưa kết quả dự đoán về giá trị ban đầu của Bitcoin
    predicted_prices_denormalized = []
    for i in range(len(predictions_5_day)):
        denormalized_price = (predictions_5_day[i]+1) * test_set['bt_Close'].values[i] 
        predicted_prices_denormalized.append(denormalized_price)
    predicted_prices_5 = np.array(predicted_prices_denormalized)
    
    result = {"predict_1_day":predicted_prices_1, "predict_5_day": predicted_prices_5}
    
    return result
    
    
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
        'X-CMC_PRO_API_KEY': '99f6663d-cd5c-4a75-8e1e-e68080bf23a5'
    }
    parameters = {
    'symbol': name
    }
    response = requests.get(url, headers=headers, params=parameters)
    data = response.json()["data"]
    coin_info = data[name]
    description = {
        "description": coin_info[0]["description"],
        "logo": coin_info[0]["logo"],
        "message_board": coin_info[0]["urls"]["message_board"][0],
        "reddit": coin_info[0]["urls"]["reddit"][0],
        "source_code": coin_info[0]["urls"]["source_code"][0],
        "technical_doc": coin_info[0]["urls"]["technical_doc"][0],
        "website": coin_info[0]["urls"]["website"][0],
        
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
    # Save df to session
    # session['df'] = data_list

    des = descripton(bitcoin_name)
    # print(des)
    result_predict = predict(data)
    
    # Lấy ngày hiện tại
    today = datetime.datetime.today()
    next_date = []
    # Hiển thị date của 5 ngày tiếp theo
    for i in range(0, 5):
        next_day = today + datetime.timedelta(days=i)
        next_date.append(next_day.date())
    
    # Tạo dataframe với 2 cột là 'date' và 'price'
    df_5_day = pd.DataFrame({'date': next_date, 'price': result_predict['predict_5_day'][0]})
    df_5_day_html = df_5_day.to_dict('records')
    # In ra dataframe
    # print(df_5_day)

    # create a figure with Bokeh
    predict_5 = figure(title="Bitcoin Price Predict",height=350, sizing_mode='stretch_width')
    predict_5.line(df_5_day['date'], df_5_day['price'], line_width=2)


    xformatter = DatetimeTickFormatter(days="%Y-%m-%d")
    predict_5.xaxis.formatter = xformatter
    # # generate the HTML code for the plot
    plot_html = file_html(predict_5, CDN)
    # Render template hiển thị kết quả
    return render_template('./index.html', data1= data_list, data2= des ,data3 = result_predict, data4 = df_5_day_html, plot1 = dashboard(bitcoin_name), plot2 = plot_html)

# @app.route('/dashboard')


if __name__ == '__main__':
    app.run(debug=True)