#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 14:08:06 2020

@author: arnav
"""

from bs4 import BeautifulSoup
import requests
import pandas


def create_directories():
    return


def find_html_tag():
    return


def download_line_list():
    return


def summon_HITEMP(molecule, isotopologue):
    return


url = 'https://hitran.org/hitemp/'

web_content = requests.get(url).text
    
soup = BeautifulSoup(web_content, "lxml")

table = soup.find('table')

n_rows = 0
n_columns = 0
column_names = []

for row in table.find_all('tr'):
    td_tags = row.find_all('td')
    if len(td_tags) > 0:
        n_rows+=1
        if n_columns == 0:
            # Set the number of columns for our table
            n_columns = len(td_tags)
                        
    # Handle column names if we find them
    th_tags = row.find_all('th') 
    if len(th_tags) > 0 and len(column_names) == 0:
        for th in th_tags:
            column_names.append(th.get_text())
  
"""          
df = pandas.DataFrame(columns = column_names, index= range(0,n_rows))


row_marker = 0
for row in table.find_all('tr'):
    column_marker = 0
    columns = row.find_all('td')
    for column in columns:
        print(column_marker)
        df.iat[row_marker,column_marker] = column.get_text()
        column_marker += 1
    row_marker += 1

"""