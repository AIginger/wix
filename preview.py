# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 18:16:21 2019

@author: HaiyanJiang
@email : jianghaiyan.cn@gmail.com

"""


def rename_docfile(from_path, rename_loger, **param):
    # param is a dict, test_station, ntype, model_series
    # 'AT_nor_laya_1.csv'
    import os
    import glob
    import pandas as pd
    pattern = '*.csv'
    flist = glob.glob(os.path.join(from_path, pattern))
    nn = len(flist)
    df_name = pd.DataFrame(index=range(nn), columns=['old_fname', 'new_fname'])
    for i in range(nn):  # i = 0
        pathAndFilename = flist[i]
        print(pathAndFilename)
        f_name = os.path.basename(pathAndFilename)
        title, ext = os.path.splitext(f_name)
        prename = '_'.join(param.values())
        new_name = prename + '_' + str(i) + ext
        os.rename(pathAndFilename, os.path.join(from_path, new_name))
        df_name.loc[i, 'old_fname'] = f_name
        df_name.loc[i, 'new_fname'] = new_name
    df_name.to_csv(rename_loger, index=False, header=True)
    print('666 DONE!')


def get_colnames(from_path, savename):
    # from_path = r'D:/ATdata2018/tmp'
    # savename = r'D:/ATdata2018/tmp_colnames.csv'
    import glob
    import pandas as pd
    from .utils import read_csv_top_k
    flist = glob.glob(from_path + '/*.csv')  # len(flist)
    df_cnames = {}
    for filename in flist:  # filename = flist[0]
        tmp = read_csv_top_k(filename, k=20, sep=',')
        col_names = list(tmp.columns)
        # print(col_names)
        df_cnames[filename] = col_names
    df = pd.DataFrame.from_dict(df_cnames, orient='index')
    df.to_csv(savename, index=True, header=True)


if __name__ == '__main__':
    from_path = r'D:/ATdata2018/tmp'
    rename_loger = r'D:/ATdata2018/tmp_rename_loger.csv'
    param = dict(test_station='AT', ntype='nor', model_series='laya')
    param = dict(t_station='nor')  # 'nor' or 'exp', which is a must
    rename_docfile(from_path, rename_loger, **param)
    savename = r'D:/ATdata2018/tmp_colnames.csv'
    get_colnames(from_path, savename)
    print("666 DONE!")
