import time
import undetected_chromedriver as uc 
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from geopy.geocoders import Nominatim

import re
import pandas as pd 
import numpy as np


def page_down(driver):
    driver.execute_script('''
        const scrollStep = 300;
        const scrollInterval = 100;

        const scrollHeight = document.documentElement.scrollHeight;
        let currentPosition = 0;
        const interval = setInterval(() => {
            window.scrollBy(0, scrollStep);
            currentPosition += scrollStep;

            if (currentPosition >= scrollHeight) {
                clearInterval(interval);
            }
        }, scrollInterval);
    ''')


def get_urls(link, pages=5):
    driver = uc.Chrome()
    driver.implicitly_wait(5)
    
    all_urls = []
    
    try:
        for page in range(1, pages + 1):
            page_link = f"{link}?page={page}"
            driver.get(url=page_link)
            time.sleep(4)
            
            page_down(driver=driver)
            time.sleep(4)
            
            find_links = driver.find_elements(By.CLASS_NAME, 'a4tiB2')
            urls = [f'{link.get_attribute("href")}' for link in find_links]
            all_urls.extend(urls)
    
    finally:
        driver.quit()

    unique_urls = list(set(all_urls))
    urls_dict = {k: v for k, v in enumerate(unique_urls)}
    
    return urls_dict


def get_info(url):
    driver = uc.Chrome()
    driver.implicitly_wait(5)
    
    try:
        driver.get(url=url)
        time.sleep(2)
        page_down(driver=driver)
        time.sleep(2)

        page_source = str(driver.page_source)
        soup = BeautifulSoup(page_source, features="lxml")

        info = {}
        
        price = soup.find_all('div', {'class': 'JfVCK'})
        info['price'] = int(price[0].text.replace('\xa0', '').replace('₽', '').replace(' ', '')) if price else None
        
        adress = soup.find_all('span', {'class': 'ItUnT', 'itemprop': 'name'})
        info['adress'] = str(adress[0].text.replace('\xa0', '').replace(' ', '')) if adress else None
        
        info_app = soup.find('section', attrs={'class': 'product-page__section', 'data-e2e-id': 'product-details'})
        if info_app:
            app_name = info_app.find_all('span', {'class': 'gqoOy'})
            app_info = info_app.find_all('span', {'class': 'ffG_w', 'data-e2e-id': 'Значение'})
            for i in range(min(len(app_name), len(app_info))):
                info[f'{app_name[i].text}'] = app_info[i].text
        else:
            print(f"Характеристики квартиры не найдены на странице: {url}")

        info_house = soup.find('section', attrs={'class': '_Xcv2', 'data-e2e-id': 'building-info-block'})
        if info_house:
            house_name = info_house.find_all('span', {'class': 'sQK5j'})
            house_info = info_house.find_all('span', {'class': 'upbHP', 'data-e2e-id': 'Значение'})
            for i in range(min(len(house_name), len(house_info))):
                info[f'{house_name[i].text}'] = house_info[i].text
        else:
            print(f"Информация о доме не найдена на странице: {url}")

        df = pd.DataFrame([info])
        return df

    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")
        return pd.DataFrame()

    finally:
        driver.quit()


def get_coordinate(address):
    geolocator = Nominatim(user_agent="Yandex", timeout=10)
    time.sleep(1) 
    try:
        location = geolocator.geocode(address)
        if location:
            return [location.latitude, location.longitude]
        else:
            return [None, None]
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        return [None, None]
    

def format_address(address):
    address = re.sub(r'улица', ' улица ', address)
    address = re.sub(r'проспект', ' проспект ', address)
    address = re.sub(r'бульвар', ' бульвар ', address)
    address = re.sub(r'переулок', ' переулок ', address)
    address = re.sub(r'шоссе', ' шоссе ', address)
    address = re.sub(r'проезд', ' проезд ', address)
    address = re.sub(r'площадь', ' площадь ', address)
    address = re.sub(r'набережная', ' набережная ', address)
    
    address = re.sub(r'\s+', ' ', address)
    
    return address.strip()

def fix_address(address):
    address = re.sub(r'([а-яё])([А-ЯЁ])', r'\1 \2', address)
    
    address = re.sub(r'(\d+[а-яА-Я]*)\S*', r'\1', address)
    
    return address


def format_address(address):
    formatted_address = re.sub(r'(\d+[а-яА-Я]*).*', r'\1', address)
    return formatted_address


def main():
    url = 'https://vladivostok.domclick.ru/pokupka/kvartiry/vtorichka'
    dict_urls = get_urls(url)

    data = []
    for i in range(len(dict_urls)):
        data_house = get_info(dict_urls[i])
        data.append(data_house)
        print(f'Обработано: {i + 1}/{len(dict_urls) + 1}')

    df = pd.concat(data, ignore_index=True)
    df['adress'] = df['adress'].str.replace(',', ', ')
    df['adress'] = df['adress'].apply(format_address)
    df['adress'] = df['adress'].apply(fix_address)
    df['adress'] = df['adress'].apply(format_address)
    df['adress'] = df['adress'].str.replace(' ,', ',')

    df['coordinates'] = df['adress'].apply(get_coordinate)
    df[['latitude', 'longitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)

    df.drop(columns=['coordinates'], inplace=True)

    df['Кухня'] = df['Кухня'].str.replace('м2', '').str.replace(',', '.').astype(float)
    df['Площадь'] = df['Площадь'].str.replace('м2', '').str.replace(',', '.').astype(float)

    columns_translation = {
        'price': 'price',  # Цена
        'adress': 'address',  # Адрес
        'Комнат': 'rooms',  # Количество комнат
        'Площадь': 'area',  # Площадь
        'Кухня': 'kitchen_area',  # Площадь кухни
        'Этаж': 'floor',  # Этаж
        'Ремонт': 'renovation',  # Тип ремонта
        'Тип сделки': 'deal_type',  # Тип сделки
        'Балкон': 'balcony',  # Балкон
        'Мусоропровод': 'trash_chute',  # Мусоропровод
        'Лет в собственности': 'years_in_ownership',  # Лет в собственности
        'Количество собственников': 'number_of_owners',  # Количество собственников
        'Класс энергоэффективности': 'energy_efficiency_class',  # Класс энергоэффективности
        'Энергоснабжение': 'power_supply',  # Энергоснабжение
        'Количество лифтов': 'number_of_elevators',  # Количество лифтов
        'Вентиляция': 'ventilation',  # Вентиляция
        'Грузовой лифт': 'freight_elevator',  # Грузовой лифт
        'Газ': 'gas_supply',  # Газоснабжение
        'latitude': 'latitude',  # Широта
        'longitude': 'longitude'  # Долгота
    }

    df.rename(columns=columns_translation, inplace=True)

    df.to_csv('data_1.csv')
    return True

if __name__ == '__data_parser__':
    main()

