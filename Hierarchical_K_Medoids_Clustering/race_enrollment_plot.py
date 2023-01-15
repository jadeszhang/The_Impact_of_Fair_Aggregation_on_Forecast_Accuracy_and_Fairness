import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Year = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
White = [13383916, 13668767, 13518208, 13099786, 12834496, 12558358, 12363717, 12079378, 11853923, 11655308]
Black = [3036408, 3104700, 3180593, 3039785, 2931768, 2844925, 2787131, 2699094, 2637397, 2611687]
Native = [155202, 153736, 156670, 135390, 135505, 137224, 127751, 136767, 136862, 148293]
Asian = [1261282, 1280963, 1315018, 1291115, 1338141, 1375113, 1371995, 1396900, 1399514, 1347417]
Pacific = [39945, 35959, 36127, 34742, 36298, 39426, 37019, 38866, 41280, 36243]
Hispanic = [2895028, 3098522, 3251736, 3271250, 3364282, 3436122, 3478364, 3551233, 3570039, 3597258]

df=pd.DataFrame({'Year': Year, 'White': White, 'Black/African American': Black, 
                 
                 'American Indian and Alaska Native': Native, 'Asian':Asian,
                 'Native Hawaiian and Other Pacific Islander':Pacific,
                 'Hispanic and Latino':Hispanic})


f, axes = plt.subplots(3, 1,figsize=(15, 15))

axes[1].plot('Year', 'Black/African American', data=df, marker='o', markersize=6, markerfacecolor='peru', color='navajowhite', linewidth=3)
axes[1].plot('Year', 'Asian', data=df, marker='o', markersize=6, markerfacecolor='tomato', color='lightpink', linewidth=3)
axes[1].plot('Year', 'Hispanic and Latino', data=df, marker='o', markersize=6, markerfacecolor='grey', color='silver', linewidth=3)


axes[0].plot(  'Year', 'Native Hawaiian and Other Pacific Islander', data=df, marker='o', markersize=6, markerfacecolor='mediumpurple', color='thistle', linewidth=3)
axes[0].plot(  'Year', 'American Indian and Alaska Native', data=df, marker='o', markersize=6, markerfacecolor='seagreen', color='palegreen', linewidth=3)


axes[2].plot( 'Year', 'White', data=df, marker='o', 
             markerfacecolor='royalblue', markersize=6, color='lightskyblue', linewidth=3)
# plt.plot( 'Year', 'White', data=df, marker='o', markerfacecolor='royalblue', markersize=6, color='lightskyblue', linewidth=3)
# plt.plot( 'Year', 'Black/African American', data=df, marker='o', markersize=6, markerfacecolor='peru', color='navajowhite', linewidth=3)
# plt.plot( 'Year', 'Asian', data=df, marker='o', markersize=6, markerfacecolor='tomato', color='lightpink', linewidth=3)

# plt.plot( 'Year', 'American Indian and Alaska Native', data=df, marker='o', markersize=6, markerfacecolor='seagreen', color='palegreen', linewidth=3)
# plt.plot( 'Year', 'Native Hawaiian and Other Pacific Islander', data=df, marker='o', markersize=6, markerfacecolor='mediumpurple', color='thistle', linewidth=3)
for i in [0,1, 2]:
    
    axes[i].ticklabel_format(style='sci', useMathText = True, axis='y', scilimits=(0,0))
    axes[i].legend(loc = 'center left')
    axes[i].grid(color = 'gainsboro', linestyle='-', linewidth=0.5)
    axes[i].set_xticks(Year) 
plt.xlabel('Year', fontsize=14)
axes[1].set_ylabel('Enrollment count', fontsize=14)
f.show()
