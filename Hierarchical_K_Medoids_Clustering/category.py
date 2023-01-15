'''This function is used to get all the rules and matches of the df
    INPUT: df (dataframe)
    OUTPUT: race_std_agg_cats (dictionarty), cat_agg_disagg_match (dictionary),
    race_disagg_cats (dictionary) '''

def category():
    from dataclasses import replace
    import pandas as pd
    import os
    import glob
    import pandas as pd
    import numpy as np
    import chardet

    
    
    
    files = os.listdir('C:/Users/jz839/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment')
    dfs = {}
    for filename in files: # read all the files
        if filename.endswith('.csv'):
            #concast the file name
            fullfilename = 'C:/Users/jz839/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/' + filename
            # check encoding
            with open(fullfilename, 'rb') as f:
                result = chardet.detect(f.read())
            f = pd.read_csv(fullfilename, encoding=result['encoding'], sep="\t", thousands=',')        
            dfs[filename] = f

    df_2017 = pd.read_csv('C:/Users/jz839/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/2017.csv', sep=",", thousands=',')
    df_2018 = pd.read_csv('C:/Users/jz839/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/2018.csv', sep=",", thousands=',')
    df_2019 = pd.read_csv('C:/Users/jz839/OneDrive/research/Forecasting Paper/Data/admission_and_enrollment/2019.csv', sep=",", thousands=',')

    dfs['2017.csv'] = df_2017
    dfs['2018.csv'] = df_2018
    dfs['2019.csv'] = df_2019

    # define the broad cats
    race_std_agg_cats = {'African American and Black': 'a_race_1', 
                        'Hispanic/Latinx': 'a_race_2',
                        'American Indian/Alaska Native': 'a_race_3',
                        'Asian': 'a_race_5',
                        'Native Hawaiian and Pacific Islander': 'a_race_4',
                        'Southwest Asian/North African': 'a_race_6',
                        'White': 'a_race_7'}

    ####### define the detailes cats (for model_2 set)
    cat_agg_disagg_match = {}
    for race in race_std_agg_cats:
        cat_agg_disagg_match[race_std_agg_cats[race]] = []

    # search all the disaggregate race cats in each dataset
    for df_name in dfs: 
        for agg_race in race_std_agg_cats:
            df = dfs[df_name]
            l_1 = list(df.loc[df['BCAT'] == agg_race]['CAT'].unique())
            cat_agg_disagg_match[race_std_agg_cats[agg_race]] = cat_agg_disagg_match[race_std_agg_cats[agg_race]] + l_1

    # clean the cat match
    for disagg_race in cat_agg_disagg_match:
        l_2 = cat_agg_disagg_match[disagg_race]
        cat_agg_disagg_match[disagg_race] = list(np.unique(np.array(l_2)))



    ####### clean every dataset (only keeps disaggregate cat)
    # initiallize list for enrollment & admission
    enrollment_dfs = {}
    admission_dfs = {}
    for df_name in dfs:
        df = dfs[df_name]
        # get the year 
        name = df_name
        name = df_name.split('.')[0]
        # keeps enrollment 
        df_enrollment = df.drop(['BCAT','APPS',"ADMITS"],axis = 1)
        enrollment_dfs[name] = df_enrollment
        # keeps admission    
        df_admission= df.drop(['BCAT','APPS',"ENROLLEES"],axis = 1)
        admission_dfs[name] = df_admission

    #### clean up the datasets
    # initialize the dfs 
    dfs_enrollment = []
    dfs_admission = []

    # for each dictionary 
    i = 1
    for d in [enrollment_dfs, admission_dfs]:
        # for each dataset 
        for year in d:
            i = i + 1
            # transpose
            df = d[year].T
            # make the first row the index
            df.columns = df.iloc[0]
            df = df.drop('CAT')
            df['year'] = year
            df = df.set_index('year')
            if (i <= 11):
                dfs_enrollment.append(df)
            else: 
                dfs_admission.append(df)

    # merge the datasets in enrollment
    df_enrollment = pd.concat(dfs_enrollment)
    df_admission = pd.concat(dfs_admission)

    # get the list of disaggregate race
    if (list(df_enrollment.columns) != list(df_admission.columns)):
        print('ATTENTION!!')
    l_race = list(df_enrollment.columns)

    # initialize directory for race and indexes
    race_disagg_cats = {}

    for i in range(0, len(l_race)):
        name = "race_" + str(i)
        race_disagg_cats[name] = l_race[i]

    return race_std_agg_cats, cat_agg_disagg_match, race_disagg_cats