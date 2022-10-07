import os
import re
import requests
import sys
import json
from bs4 import BeautifulSoup
import ddddocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import six
import csv


#設定matplotlib字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

#宣告OCR物件
ocr = ddddocr.DdddOcr()

#重複執行直到驗證通過
while True:
    #宣告Session物件
    session = requests.Session()
    #設定目標網址
    resp = session.get('https://bsr.twse.com.tw/bshtm/bsMenu.aspx', verify=False)

    #如果回應200代表請求成功
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, 'lxml')
        nodes = soup.select('form input')
        params = {}
        for node in nodes:
            name = node.attrs['name']

            if name in ('RadioButton_Excd', 'Button_Reset'):
                continue

            if 'value' in node.attrs:
                params[node.attrs['name']] = node.attrs['value']
            else:
                params[node.attrs['name']] = ''

        captcha_image = soup.select('#Panel_bshtm img')[0]['src']
        m = re.search(r'guid=(.+)', captcha_image)
        if m is None:
            exit(1)

        imgpath = '%s.jpg' % m.group(1)
        url = 'https://bsr.twse.com.tw/bshtm/' + captcha_image
        resp = requests.get(url, verify=False)
        if resp.status_code == 200:
            with open(imgpath, 'wb') as f:
                f.write(resp.content)
            os.system('open ' + imgpath)
            f.close()

        with open(imgpath, 'rb') as f:
            img_bytes = f.read()
        #利用OCR辨識圖片
        vcode = ocr.classification(img_bytes)
        f.close()
        try:
            os.remove(imgpath)
        except():
            pass
        params['CaptchaControl1'] = vcode
        #股票代碼
        params['TextBox_Stkno'] = '2317'
        print(json.dumps(params, indent=2))
        #送出資料
        resp = session.post('https://bsr.twse.com.tw/bshtm/bsMenu.aspx', data=params)
        if resp.status_code != 200:
            print('任務失敗: %d' % resp.status_code)
            continue
        #利用BeautifulSoup解析HTML
        soup = BeautifulSoup(resp.text, 'lxml')
        nodes = soup.select('#HyperLink_DownloadCSV')
        if len(nodes) == 0:
            print('任務失敗，沒有下載連結')
            continue

        #下載分點進出CSV
        resp = session.get('https://bsr.twse.com.tw/bshtm/bsContent.aspx')
        with open('output.csv', 'w', encoding="utf-8") as f:
            writer = csv.writer(f)
            for resp_line in resp.text.split('\n')[2:]:
                a = resp_line.split(',')
                print(a)
                if len(a) >= 4:
                    a[1] = re.sub(r'[0-9A-Za-z ]+', '', a[1])
                    if len(a) >= 10:
                        a[7] = re.sub(r'[0-9A-Za-z ]+', '', a[7])
                    writer.writerow(a)
            f.close()

        if resp.status_code != 200:
            print('任務失敗，無法下載分點進出 CSV')
            continue
        break

# 兩欄合併成一欄
data = pd.read_csv("output.csv")
print(data)
combine = pd.DataFrame()
combine['券商'] = pd.concat([data['券商'], data['券商.1']], axis=0)
combine['價格'] = pd.concat([data['價格'], data['價格.1']], axis=0)
combine['買進股數'] = pd.concat([data['買進股數'], data['買進股數.1']], axis=0)
combine['賣出股數'] = pd.concat([data['賣出股數'], data['賣出股數\r']], axis=0)
combine = combine.sort_values(["券商"], ascending=True, ignore_index=True)
combine = combine.dropna(axis=0, how='any')
combine['價格'] = combine['價格'].astype(float)
combine['買進股數'] = combine['買進股數'].astype(float)
combine['賣出股數'] = combine['賣出股數'].astype(float)
print(combine)

# 針對買進、賣出、買超、均買價、均賣價進行計算
temp = pd.DataFrame()
temp["買進"] = combine.groupby(by='券商').apply(lambda x: (x['買進股數']/1000).sum())
temp["賣出"] = combine.groupby(by='券商').apply(lambda x: (x['賣出股數']/1000).sum())
temp["買超"] = combine.groupby(by='券商').apply(lambda x: ((x['買進股數'] - x['賣出股數'])/1000).sum())
temp["均買價"] = combine.groupby(by='券商').apply(lambda x: (x['買進股數']/1000 * x['價格']).sum()/(x['買進股數']/1000).sum())
temp["均賣價"] = combine.groupby(by='券商').apply(lambda x: (x['賣出股數']/1000 * x['價格']).sum()/(x['賣出股數']/1000).sum())


# 依照買超排序，並取出兩表(買超為正和買超為負的兩表)
temp = temp.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
positive = temp[temp['買超'] >= 0]
positive = positive.sort_values(by='買超', ascending=False).reset_index()
negative = temp[temp['買超'] < 0]
negative = negative.sort_values(by='買超').reset_index()
del positive['index']
del negative['index']

# 將兩表合併，並儲存至csv
result = pd.concat([positive, negative], axis=1).round(2)
# result.set_axis(['券商', '買進', '賣出', '買超', '均買價', '均賣價']*2, axis=1, inplace=True)
print(result)
result.to_csv("result.csv")
