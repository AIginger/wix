# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:01:03 2019

@author: HaiyanJiang
@email : jianghaiyan.cn@gmail.com

"""

from wix import rename_docfile, get_colnames
from wix import WIXtable
from wix import CleanXdata
from wix import ConstructXYdata


print(__doc__)


def rename_():
    from_path = r'D:/ATdata2018/tmp'
    rename_loger = r'D:/ATdata2018/tmp_rename_loger.csv'
    param = dict(test_station='AT', ntype='nor', model_series='laya')
    param = dict(t_station='nor')  # 'nor' or 'exp', which is a must
    rename_docfile(from_path, rename_loger, **param)
    savename = r'D:/ATdata2018/tmp_colnames.csv'
    get_colnames(from_path, savename)
    print("666 DONE!")


def get_xtable():
    # the final result data saves in data_path + 'XtableR3Fold'
    sel_names = [
        'storetime', 'sn', 'uutname', 'wsname', 'r2testitemname',
        'r3subtestitemname', 'r3result', 'r3resultdesc']
    s_dtype = {
        'storetime': str,
        'sn': str,
        'uutname': str,
        'wsname': str,
        'r2testitemname': str,
        'r3subtestitemname': str,
        'r3result': float,
        # 'r3resultdesc': float
        }
    keep_names = ['sn', 'r2testitemname', 'r3subtestitemname']
    rename_dict = {
        'sn': 'SERIAL_NUMBER',
        'r2testitemname': 'R2_FT_TEST_ITEM_NAME',
        'r3subtestitemname': 'R3_FT_TEST_ITEM_NAME',
        'r3resultdesc': 'RESULT_DESC',
        'wsname': 'STATION_NAME',
        'storetime': 'TEST_DATE',
        'uutname': 'UUT_NAME',
        'r3result': 'R3_RESULT'}
    gb_names = ['SERIAL_NUMBER', 'TEST_DATE',  'UUT_NAME', 'R3_RESULT']
    fold_cols = {'R3': ['R3_RESULT'], 'Date': ['TEST_DATE']}
    sub_docs = {'R3': 'XtableTMP', 'Date': 'XtableR3Fold'}
    # gb_names = ['SERIAL_NUMBER']
    # fold_cols = {}
    # sub_docs = {}
    from_path = r'D:/ATdata2018/tmp'
    data_path = r'D:/HWtmp2019'
    wxt = WIXtable(
        from_path, data_path, sel_names, s_dtype, keep_names, rename_dict,
        gb_names, fold_cols, sub_docs)
    wxt.wixtable()
    print("666 DONE!")


def get_xclean():
    # clean the X data
    # data_path = r'D:/HWtmp2019'
    # docname = 'XtableNumFold'
    data_path = r'D:/HWtmp2019'
    docname = 'XtableDateFold'
    rm_nanratio=0.2; rm_uniqnum=10; rm_snratio=0.01
    cx = CleanXdata(data_path, docname, rm_nanratio, rm_uniqnum, rm_snratio)
    cx.xclean()
    print("666 DONE!")


def get_xydata():
    # clean the X data
    # data_path = r'D:/HWtmp2019'
    # docname = 'XtableNumFold'
    data_path = r'D:/HWtmp2019'
    doc_from = 'XtableDateFold'
    doc_tmp = 'XY01data'
    doc_to = 'XYdata'
    cx = ConstructXYdata(data_path, doc_from, doc_tmp, doc_to)
    cx.xydata()
    print("666 DONE!")


if __name__ == '__main__':
    # rename_()
    # get_wide_xtable()
    # get_xclean()
    # get_xydata()
    '''
    rename_()
    get_xtable()
    get_xclean()
    get_xydata()
    '''
    print("666 DONE!")





