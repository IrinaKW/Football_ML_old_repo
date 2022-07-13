#Collect ELO from the links

#%%
import pandas as pd 
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import logging
import os
import sys
import time

import config


# %%
#Create scraper
class soccer_ELO_scraper:
    '''
    Extract ELO data per match for two teams
    The dataframe with links is provided.
    Scraper gets the ELO rank and add it to the dataframe
    '''

    def __init__(self):
        link2= 'https://aicore-files.s3.amazonaws.com/Data-Science/Match_Info.csv'
        self.match_link = pd.read_csv(link2)
        pd.set_option('display.max_columns', None) 
        self.match_link.drop(['Date_New','Referee','Home_Yellow','Away_Yellow','Home_Red','Away_Red'],inplace=True, axis=1)
        self.match_link['Link']=self.match_link['Link'].astype('string')
        self.match_link['Link']='https://www.besoccer.com'+self.match_link['Link']
        self.match_link['Home_ELO']=''
        self.match_link['Away_ELO']=''

    
    def set_up(self):
        if not sys.warnoptions:
            import warnings
            warnings.simplefilter("ignore")
        logging.getLogger('WDM').setLevel(logging.NOTSET)
        os.environ['WDM_LOG'] = "false"
        
        chrome_options = Options()
        #chrome_options.add_argument("--disable-gpu")
        #chrome_options.add_argument('--headless')
        #chrome_options.add_argument('--no-sandbox')
        #chrome_options.add_argument('--disable-dev-shm-usage')
        #chrome_options.add_argument('--window-size=1920,1080')
        #chrome_options.add_argument('--remote-debugging-port=9222')
        self.driver = webdriver.Chrome(options=chrome_options) #(for local run)
        #self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
        link='https://www.besoccer.com/match/saarbrucken/stuttgarter-kickers/19903487'
        self.driver = webdriver.Chrome()
        self.driver.get(link)
        time.sleep(10)

    def tearDown(self):
        self.driver.quit()

    
    def cookies(self):
        """Open URL and identifies the cookies button on the page and click on it.
        Attr:
            driver (interface); the chrome webdriver used by selenium to control the webpage remotely 
        Raises:
            pass if there is no cookies button but has the URL open
         """

        try: 
            accept_cookies_button = self.driver.find_element(By.XPATH, config.COOKIE_XPATH)
            accept_cookies_button.click()
            time.sleep(2)
        except:
            pass # If there is no cookies button, we won't find it, so we can pass


    def subscribe(self):
        """Open URL and identifies the "not to subscribe" button on the page and click on it.
        Attr:
            link (str): the global variable, link to the match
            driver (interface); the chrome webdriver used by selenium to control the webpage remotely 
        Raises:
            pass if there is no "no subscribe" button but has the URL open
         """

        try: 
            popup_element = self.driver.execute_script("return document.querySelector('#match > div.grv-dialog-host').shadowRoot.querySelector('#grv-popup__subscribe')")
            popup_element.click()
            time.sleep(2)
        except:
            pass # If there is no subcribe pop-up, we won't find it, so we can pass

    
    def scraping(self):
        for i in range(3): #(len(self.match_link.index)):
            link=str(self.match_link.loc[i, 'Link'])
            print(link)
            self.driver.get(link)
            time.sleep(5)
            self.driver.find_element(By.XPATH, config.ANALYSIS_XPATH).click()
            time.sleep(2)
            home_elo=self.driver.find_element(By.XPATH, config.HOME_ELO_XPATH).text
            away_elo=self.driver.find_element(By.XPATH, config.AWAY_ELO_XPATH).text
            self.match_link.loc[i, ['Home_ELO']]=home_elo
            self.match_link.loc[i, ['Away_ELO']]=away_elo
            time.sleep(2)
        self.match_link.to_csv('match_ELO.csv')


    
if __name__ == "__main__":
        #initiate the class
    scraper= soccer_ELO_scraper()
    scraper.set_up()
    scraper.subscribe()
    scraper.cookies()
    scraper.scraping()
    scraper.tearDown()

