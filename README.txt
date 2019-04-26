
Created on Thu Apr 18 22:57:29 2019

INTRO:
must check the step0-step3 carfully, then adjust and set the proper parameters
for the functions.



step0 - preview of the raw files
(details of how to set parameters is in SET_PARAM.txt and READMEwix.txt)
(params need to be modified according to the specific problem)

from_path: the raw data path, must be a full path name, not the absolute path.
rename_loger: to rename .csv in 'from_path', must be a full path.
savename: save the colnames of all the *.csv from 'from_path'.
param: dict(t_station='nor')  # 'nor' or 'exp', which is a must

POTENTIAL ERRORS:
if the *.csv in 'from_path' donnot share the same column names, there might
pop up errors.

EXAMPLES:
from_path = r'D:/ATdata2018/tmp'
rename_loger = r'D:/ATdata2018/tmp_rename_loger.csv'
param = dict(test_station='AT', ntype='nor', model_series='laya')  # perfect
param = dict(t_station='nor')  # 'nor' or 'exp', which is a must
savename = r'D:/ATdata2018/tmp_colnames.csv'


step1 - set proper parameters
set proper parameters so that it can be used in converting to wide Xtable

NOTES: the sel_names, s_dtype, rename_dict must include the same names,
except that there is no 'r3resultdesc' in stype (because 'r3resultdesc'
might be 'str' or 'float', we donot know exactly what type it is)

valid_params:
sel_names:
    list, selected columns in the 'from_path', where the following 5 must be inclued
    ['sn', 'r2testitemname', 'r3subtestitemname', 'r3resultdesc', 'wsname']
s_dtype:
    dict, specifying the format of the data, must be a corresponding dictionary
    to sel_names and we can skip 'r3resultdesc'.
rename_dict:
    dict, replace the old column name by the new one, must be a corresponding
    dictionary to sel_names.
keep_names:
    list, must have at least ['sn', 'r2testitemname', 'r3subtestitemname']
gb_names: list, groupby names, ['SERIAL_NUMBER', 'TEST_DATE',  'UUT_NAME', 'R3_RESULT'],
fold_cols: dict, keyword and a one length list.
sub_docs: dict, data used for folding and data saved file for each document


EXAMPLES: parameter usage refer to SET_PARAM.txt
EXAMPLES: function usage refer to wix_exsample.py



step2 - we have an example for step0 and step1 (wix_exsample.py)

parameters meaning:
from_path: the raw data path, must be a full path name, not the absolute path.
data_path: the data path where we want to save the data

EXAMPLES:
from_path = r'D:/ATdata2018/tmp'
data_path = r'D:/HWtmp2019'



NOTES:
1. xtable_fold() is a must if the information we want include
{'storetime', 'uutname', 'r3result'}. Then set carefully of parameters
'gb_names', 'fold_cols', 'sub_docs'



2. the docment-chain in 'sub_docs' works like this:
'XtableTMP' -> 'XtableR3Fold' -> 'XtableDateFold'

EXAMPLE:
gb_names = ['SERIAL_NUMBER', 'TEST_DATE',  'UUT_NAME', 'R3_RESULT']
fold_cols = {'R3': ['R3_RESULT'], 'Date': ['TEST_DATE']}
sub_docs = {'R3': 'XtableTMP', 'Date': 'XtableR3Fold'}

which means,
when we run into 'R3', we want to fold the column of ['R3_RESULT'],
we use datafrom 'XtableTMP', and the result data will be saved in 'XtableR3Fold';
when we run into 'Date', we want to fold the column of  ['TEST_DATE'],
we use datafrom 'XtableR3Fold', and the result data will be saved in 'XtableDateFold'.

docment-chain is 'XtableTMP' -> 'XtableR3Fold' -> 'XtableDateFold'

'XtableDateFold' keeps the wide xtable we need.



NOTE: logers files (LOGS of the data_path)
we have keep some loger files to check if the function is proper executed, and
record the time it uses. They will be autosaved in -- data_path + 'LOGS'
(686M csv needs 664s)











