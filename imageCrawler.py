#최종 수정일 : 2024.03.05
#수정 : jpg에서 png로 다운받는 형식 변경

#구글을 이용한 이미지 크롤러
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import urllib.request

# 색상 변경 시 바꿀 부분
imgAmount = 200 # 이미지 개수
imgAddress = "/Users/gimchaeu/Desktop/알약 과제/imgs_3/"
imgColor = "Black"
imgObject = " Car"
imgType = " png"
imgSlash = "/"
imgPng = ".png"

searchKey = imgColor + imgObject + imgType# 검색어

driver = webdriver.Chrome()
driver.get("https://www.google.co.kr/imghp?hl=ko&tab=wi&authuser=0&ogbl")
elem = driver.find_element("name", "q")

elem.send_keys(searchKey)
elem.send_keys(Keys.RETURN)

SCROLL_PAUSE_TIME = 1
# 스크롤 설정
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    # 가장 밑으로 스크롤
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # 페이지 로딩 대기
    time.sleep(SCROLL_PAUSE_TIME)
    # 가장 최근 스크롤 위치와 현재 스크롤 위치 비교
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        try:
            driver.find_element(By.CSS_SELECTOR, ".mye4qd").click()
        except:
            break
    last_height = new_height

images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
count = 1
for image in images:
    try:
        if count == imgAmount + 1:
            break
        image.click()
        time.sleep(0.5)
        
        imgUrl = driver.find_element(
            By.XPATH,
            '//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]'
        ).get_attribute("src")

        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent',
             'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')
        ]

        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(imgUrl, f'./imgs_3/{imgColor}/{searchKey}{str(count)}.png')
        count = count + 1
    except Exception as e:
        print('e : ', e)
        pass
driver.close()