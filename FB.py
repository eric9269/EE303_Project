from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import time
from bs4 import BeautifulSoup
import requests
import pandas
import numpy as np


def ScrapyingFB():
    # 個人資訊
    url = "https://www.facebook.com/"
    email = input("請輸入Facebook帳號")
    password = input("請輸入Facebook密碼")

    # 使用ChromeDriverManager自動下載chromedriver
    chrome_options = webdriver.ChromeOptions()
    prefs = {
        "profile.default_content_setting_values.notifications": 2
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # 使用ChromeDriverManager自動下載chromedriver
    driver = webdriver.Chrome(
        ChromeDriverManager().install(), chrome_options=chrome_options)

    # 最大化視窗
    driver.maximize_window()
    # 進入Facebook登入畫面
    driver.get(url)

    # 填入帳號密碼並送出
    driver.find_element("id", "email").send_keys(email)
    driver.find_element("id", "pass").send_keys(password)
    driver.find_element("name", "login").click()

    time.sleep(2)

    # 進入FB社團成員頁面
    driver.get("https://www.facebook.com/groups/189644259942426/members")

    time.sleep(5)

    # 往下滑5次，讓Facebook載入文章內容
    for x in range(5):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        print("scroll")
        time.sleep(1)

    soup = BeautifulSoup(driver.page_source, 'html.parser')

    titles = soup.find_all("a", href=True, class_="qi72231t nu7423ey n3hqoq4p r86q59rh b3qcqh3k fq87ekyn bdao358l fsf7x5fv rse6dlih s5oniofx m8h3af8h l7ghb35v kjdc1dyq kmwttqpk srn514ro oxkhqvkx rl78xhln nch0832m cr00lzj9 rn8ck1ys s3jn8y49 icdlwmnq cxfqmxzd pbevjfx6 innypi6y")
    names = []
    dict = {}
    for title in titles:
        # 定位每個人的名字
        print(title['href'].split("/")[4])
        posts = title.text
        names.append(posts)
        # 如果有內容才印出
        if len(posts):
            print(posts)
        # 建立姓名對映FB ID的字典
        dict[posts] = title['href'].split("/")[4]
        print("-" * 30)

    with open("FB_name.txt", encoding="utf-8", mode="w+") as f:
        for name in names:
            f.write(name)
            f.write('\n')
        f.close()
    # 關閉瀏覽器
    driver.quit()
    return dict


#建立每個人的姓名以及ID的資料
name_id_pair = ScrapyingFB()
# 讀取班級名冊
df = pandas.read_csv("1111_EE5327701_5641.csv", encoding="big5")
print(df)
name_list = np.array(df["姓名"].values)
id_list = np.array([None]*len(df["姓名"].values))
# 找尋班級名側與社團成員的交集，並將FB連結放入初始班級名冊
with open("FB_name.txt", encoding="utf-8", mode="r") as f:
    data = f.readlines()
    name = []
    for index in range(1, len(data)):
        data[index] = data[index].strip()
        if data[index] in name_list:
            id_list[np.where(name_list == data[index])] = f'https://www.facebook.com/{name_id_pair[data[index]]}'
    df['fb連結'] = id_list
    f.close()
# 存擋
df.to_csv("20220908DA課程學生資料.csv", encoding="utf_8_sig")
print(df)
