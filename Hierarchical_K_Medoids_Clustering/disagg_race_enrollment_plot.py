import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Year = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
Chinese = [1223,1432,1312,1585,1725,1585,1832,1931,1825,1955]
Puerto_Rican = [28,10,11,13,13,17,13,19,18,13]
White_Caucasian = [2107,2395,2016,2124,2416,2477,2856,2900,2509,2616]
Mexican = [541,599,666,701,887,829,1003,1085,1059,1155]
Tongan = [0,4,1,1,2,5,2,1,1,3]
Black = [160,182,170,174,196,190,212,236,226,243]

df=pd.DataFrame({'Year': Year, 'White/Caucasian': White_Caucasian, 'Black/African American': Black, 
                 
                 'Chinese': Chinese, 'Mexican/Mexican American':Mexican,
                 'Tongan':Tongan,
                 'Sri Lankan':Puerto_Rican})

plt.figure(figsize=(12, 8), dpi=80)

plt.plot('Year', 'Black/African American', data=df, marker='o', markersize=6, markerfacecolor='peru', color='navajowhite', linewidth=3)
plt.plot('Year', 'Sri Lankan', data=df, marker='o', markersize=6, markerfacecolor='tomato', color='lightpink', linewidth=3)
plt.plot('Year', 'Chinese', data=df, marker='o', markersize=6, markerfacecolor='grey', color='silver', linewidth=3)


plt.plot(  'Year', 'Mexican/Mexican American', data=df, marker='o', markersize=6, markerfacecolor='mediumpurple', color='thistle', linewidth=3)
plt.plot(  'Year', 'White/Caucasian', data=df, marker='o', markersize=6, markerfacecolor='seagreen', color='palegreen', linewidth=3)


plt.plot( 'Year', 'Tongan', data=df, marker='o', 
             markerfacecolor='royalblue', markersize=6, color='lightblue', linewidth=3)
# plt.plot( 'Year', 'White', data=df, marker='o', markerfacecolor='royalblue', markersize=6, color='l

plt.xlabel('Year', fontsize=14)
plt.ylabel('Enrollment count', fontsize=14)
plt.legend()
plt.grid(color = 'gainsboro', linestyle='-', linewidth=0.5)