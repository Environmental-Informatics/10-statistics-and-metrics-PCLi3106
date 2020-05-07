#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script servesa as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
"""
Modfied on 2020/04/28 by Pin-Ching Li
This script is written for annual and monthly analysis of streamflow dataset
Also, the data quality is checked and nan values are replaced
"""
import pandas as pd
import scipy.stats as stats
import numpy as np
from datetime import datetime, timedelta

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    # find the index of negative discharge data
    Negative_index= DataDF[DataDF["Discharge"]<0].index
    # drop those data as a gross error
    DataDF.drop(Negative_index, inplace=True)
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    # transfer start date and end date into datetime format
    D_start =datetime.strptime(startDate, '%Y-%m-%d')
    D_end   =datetime.strptime(endDate,'%Y-%m-%d')
    # find the index of negative discharge data
    less_index  = DataDF[DataDF.index<D_start].index
    larger_index= DataDF[DataDF.index>D_end].index
    # drop the data points with date before startdate
    DataDF.drop(less_index, inplace=True)
    DataDF.drop(larger_index,inplace=True)
    # report missing values
    MissingValues = DataDF['Discharge'].isna().sum()
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    # drop nan value
    Qvalues = Qvalues.dropna()
    # get Tqmean
    Tqmean  = (Qvalues>Qvalues.mean()).sum()/len(Qvalues)
    
    return (Tqmean)

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""
    # drop nan value
    Qvalues = Qvalues.dropna()
    # get RBindex
    RBindex = Qvalues.diff()
    RBindex = (abs(RBindex).sum())/(Qvalues.sum())

    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
    # drop nan value
    Qvalues = Qvalues.dropna()
    # apply rolling function with 7 days window
    val7Q   = (Qvalues.rolling(window=7).mean()).min()
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
    # drop nan
    Qvalues = Qvalues.dropna()
    median3x = (Qvalues > (Qvalues.median())*3).sum()
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    # column_names:
    colname = ['site_no','Mean Flow','Peak Flow','Median Flow',
               'Coeff Var', 'Skew', 'Tqmean', 'R-B index','7Q',
               '3xMedian']
    # resample dataframe first with water year
    DataDF_WY = DataDF.resample('AS-OCT')
    WY_avg    = DataDF_WY.mean()
    WYDataDF = pd.DataFrame(index=WY_avg.index,columns=colname)
    # get site no
    WYDataDF['site_no'] = DataDF_WY['site_no'].median()
    # get annual mean
    WYDataDF['Mean Flow'] = DataDF_WY['Discharge'].mean()
    # get annual max
    WYDataDF['Peak Flow']   = DataDF_WY['Discharge'].max()
    # get annual median
    WYDataDF['Median Flow'] =DataDF_WY['Discharge'].median()
    # get Coeff Var
    WYDataDF['Coeff Var']  = (DataDF_WY['Discharge'].std()/(DataDF_WY['Discharge'].mean()))*100
    # get skewness
    WYDataDF['Skew']   =DataDF_WY.apply({'Discharge': lambda x: stats.skew(x)})
    # get T-Q mean
    WYDataDF['Tqmean'] =DataDF_WY.apply({'Discharge': lambda x: CalcTqmean(x)})
    # get RB index
    WYDataDF['R-B index'] =DataDF_WY.apply({'Discharge': lambda x: CalcRBindex(x)})
    # get 7-day low flow
    WYDataDF['7Q'] =DataDF_WY.apply({'Discharge': lambda x: Calc7Q(x)})
    # get 3xMedian
    WYDataDF['3xMedian'] =DataDF_WY.apply({'Discharge': lambda x: CalcExceed3TimesMedian(x)})
    
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    colname = ['site_no','Mean Flow','Coeff Var','Tqmean', 'R-B index']
    # resample dataframe first with water year
    DataDF_M = DataDF.resample('MS')
    M_avg    = DataDF_M.mean()
    MoDataDF = pd.DataFrame(index=M_avg.index,columns=colname)
    # get site no
    MoDataDF['site_no'] = DataDF_M['site_no'].median()
    # get annual mean
    MoDataDF['Mean Flow'] = DataDF_M['Discharge'].mean()
    # get Coeff Var
    MoDataDF['Coeff Var']  = (DataDF_M['Discharge'].std()/(DataDF_M['Discharge'].mean()))*100
    # get T-Q mean
    MoDataDF['Tqmean'] =DataDF_M.apply({'Discharge': lambda x: CalcTqmean(x)})
    # get RB index
    MoDataDF['R-B index'] =DataDF_M.apply({'Discharge': lambda x: CalcRBindex(x)})    

    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    AnnualAverages = WYDataDF.mean()
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    Mon_avg = []
    for i in range(12):
        Mon_avg.append(MoDataDF.iloc[i:600:12].mean())
    MonthlyAverages = pd.DataFrame(Mon_avg,index=[10,11,12,1,2,3,4,5,6,7,8,9])
    MonthlyAverages = MonthlyAverages.sort_index()
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    # create output lists
    AM_out = []
    MM_out = []
    AA_out = []
    MA_out = []
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        
        # output annual and monthly metrics tables
        # store annual metrics
        WYDataDF[file]['Station']=file
        AM_out.append(WYDataDF[file])
        # store monthly metrics
        MoDataDF[file]['Station']=file
        MM_out.append(MoDataDF[file])
        
        # store annual average
        AnnualAverages[file]['Station']=file
        AA_out.append(AnnualAverages[file])
        # store monthly average
        MonthlyAverages[file]['Station']=file
        MA_out.append(MonthlyAverages[file])

    # output all the files
    # output Annual Metrics
    f_AM = open('Annual_Metrics.csv', 'w')
    i = 0
    for AM in AM_out:
        if i >0:
              f_AM = open('Annual_Metrics.csv', 'a')  
        AM.to_csv(f_AM)
        i+=1
    f_AM.close()
    # output Monthly Metrics
    f_MM = open('Monthly_Metrics.csv', 'w')
    i = 0
    for MM in MM_out:
        if i>0:
            f_MM = open('Monthly_Metrics.csv', 'a')
        MM.to_csv(f_MM)
        i+=1
    f_MM.close()    
    # output Averaged Annual Metrics
    f_AA = open('Averaged_Annual_Metrics.txt', 'w')
    i=0
    for AA in AA_out:
        if i >0:
            f_AA = open('Averaged_Annual_Metrics.txt', 'a')
        AA.to_csv(f_AA,sep='\t')
        i+=1
    f_AA.close()  
    # output Averaged Monthly Metrics
    f_MA = open('Averaged_Monthly_Metrics.txt', 'w')
    i = 0
    for MA in MA_out:
        if i>0:
            f_MA = open('Averaged_Monthly_Metrics.txt', 'a')
        MA.to_csv(f_MA,sep='\t')
        i+=1
    f_MA.close()