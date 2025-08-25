import pandas as pd
from bs4 import BeautifulSoup
import json
import requests
import re
import time
from tqdm import tqdm

sheet_names = [
               '圖書・文具・影音・樂器',
               ]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
pages = 3
urls = []
page = 1
pro_types = []
data = pd.DataFrame()
for sheet_name in sheet_names:
    df = pd.DataFrame()
    data = pd.read_excel('query.xlsx', sheet_name=sheet_name)
    data = data[data['組別'] == 26]
    queries = data['搜尋詞'].values
    for keyword in queries:
        urls.clear()
        for page in range(1, pages):
            url = 'https://m.momoshop.com.tw/search.momo?_advFirst=N&_advCp=N&curPage={}&searchKeyword={}'.format(page,
                                                                                                                  keyword)
            print(url)
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, features="lxml")
                for item in soup.select('li.goodsItemLi > a'):
                    urls.append('https://m.momoshop.com.tw' + item['href'])
            else:
                print("沒有資料")
                break
            urls = list(set(urls))
            print(len(urls))

        for i, url in enumerate(tqdm(urls)):
            columns = []
            values = []
            try:
                resp = requests.get(url, headers=headers, timeout=3)
            except:
                continue

            soup = BeautifulSoup(resp.text, features="lxml")
            # 標題
            try:
                title = soup.find('meta', {'property': 'og:title'})['content']
            except:
                continue
            if not title:
                continue
            # 品牌
            try:
                brand = soup.find('meta', {'property': 'product:brand'})['content']
            except:
                continue
            # 連結
            link = soup.find('meta', {'property': 'og:url'})['content']
            # 原價
            try:
                price = re.sub(r'\r\n| ', '', soup.find('del').text)
            except:
                price = ''
            # 特價
            amount = soup.find('meta', {'property': 'product:price:amount'})['content']
            # 類型
            cate = ''.join([i.text for i in soup.findAll('article', {'class': 'pathArea'})])
            cate = re.sub('\n|\xa0', ' ', cate)
            # 描述
            try:
                desc = soup.find('div', {'class': 'Area101'}).text
                desc = re.sub('\r|\n| ', '', desc)
            except:
                desc = ''

            columns += ['title', 'brand', 'link', 'price', 'amount', 'cate', 'desc']
            values += [title, brand, link, price, amount, cate, desc]

            # 規格
            for i in soup.select('div.attributesArea > table > tr'):
                try:
                    column = i.find('th').text
                    column = re.sub('\n|\r| ', '', column)
                    value = ''.join([j.text for j in i.findAll('li')])
                    value = re.sub('\n|\r| ', '', value)
                    columns.append(column)
                    values.append(value)
                except:
                    pass
            ndf = pd.DataFrame(data=values, index=columns).T
            try:
                df = pd.concat([df, ndf], ignore_index=True)
                print(df.shape)
            except:
                print("error")

    df.to_csv(f'./M11107505_李旭清_{sheet_name}_作業一.csv')
