import pandas as pd
from bs4 import BeautifulSoup
import json
import requests
import re
import time
from tqdm import tqdm


headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'}
keyword = '運動鞋'
pages = 2
urls = []
page = 1
for page in range(1, pages):
    url = 'https://m.momoshop.com.tw/search.momo?_advFirst=N&_advCp=N&curPage={}&searchType=1&cateLevel=2&ent=k&searchKeyword={}&_advThreeHours=N&_isFuzzy=0&_imgSH=fourCardType'.format(page, keyword)
    print(url)
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        soup = BeautifulSoup(resp.text, features="lxml")
        for item in soup.select('li.goodsItemLi > a'):
            urls.append('https://m.momoshop.com.tw'+item['href'])
    else:
        print("沒有資料")
        break
    urls = list(set(urls))
    print(len(urls))

df = pd.DataFrame()
for i, url in enumerate(tqdm(urls)):
    columns = []
    values = []

    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, features="lxml")
    # 標題
    title = soup.find('meta', {'property': 'og:title'})['content']
    # 品牌
    brand = soup.find('meta', {'property': 'product:brand'})['content']
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

    # print('==================  {}  =================='.format(i))
    # print(title)
    # print(brand)
    # print(link)
    # print(amount)
    # print(cate)

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
    df = pd.concat([df, ndf], ignore_index=True)

print(df)


local_time = time.localtime(time.time())
year = local_time.tm_year
month = local_time.tm_mon
day = local_time.tm_mday
df.to_excel(f'./MOMO{year}_{month}_{day}.xlsx')
