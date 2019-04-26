# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:38:28 2019

@author: HaiyanJiang
@email : jianghaiyan.cn@gmail.com

"""


def can_str2float(x, exclude_char='-'):
    # flag = 0 means can not convert str2float
    # flag = 1 means can convert str2float
    xu = set(x.dropna())
    if xu == set(exclude_char):
        flag = 0
        return flag
    flag = 1
    for element in xu:
        try:
            float(element)
        except ValueError:
            if element != exclude_char:
                flag = 0
                return flag
    return flag


def convert_str2float(x, exclude_char='-'):
    import numpy as np
    x_new = []
    for element in x:
        if element == exclude_char:
            v = np.nan
        else:
            v = float(element)
        x_new.append(v)
    return x_new


def data_obj_convert(df, obj_names):
    '''
    j = 'DBC_100125'  # '3.8V'  # flag=0, len(set(x)) = 1
    j = 'DBC_100131'  # '0.2A'  # flag=1, len(set(x)) > 1
    df = data
    obj_names = obj_names_1
    len(obj_names)
    aa = df[obj_names]; bb = aa[:100]
    cc = df[:40]
    j = 'DBC_100094' # contents is same as SERIAL_NUMBER 5T3JPM1674011677
    '''
    import pandas as pd
    data_new = dict()
    fs_names = []
    for j in obj_names:
        # print(j); break
        x = df[j]
        flag = can_str2float(x, exclude_char='-')
        if flag:
            x_new = convert_str2float(x, exclude_char='-')
            data_new[j] = x_new
            fs_names.append(j)
    # special_col = ['DBC_100125', 'DBC_100131']
    for j in obj_names:  # j = 'DBC_100124'  # j = 'DBC_100130'
        # print(j); break  # '3.801900V'   # just remove the last char
        x = pd.Series([v[:-1] if isinstance(v, str) else v for v in df[j]])
        flag = can_str2float(x, exclude_char='-')
        if flag:
            x_new = convert_str2float(x, exclude_char='-')
            data_new[j] = x_new
            fs_names.append(j)
    df_new = pd.DataFrame(data_new)
    return df_new, fs_names


def rep_illegal_char(text):
    # text = "(condition1) and --condition2--;;;;;::::""*000"
    import re
    sv = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    rep = {s: "_" for s in sv}
    # use these three lines to do the replacement
    rep = dict((re.escape(k), v) for k, v in rep.items())
    patn = re.compile('|'.join(rep.keys()))
    text_new = patn.sub(lambda m: rep[re.escape(m.group(0))], text)
    return text_new


def X_data_clean(data_from, filename, data, rm_nanratio=0.2, rm_uniqnum=10,
                 rm_snratio=0.01):
    '''
    data_from = r'D:/HWtmp2019/XtableDateFold'
    rm_nanratio: a float number, between 0 and 1, rm_nanratio=0.2,
        if a feature nan ratio is bigger than rm_nanratio, remove this feature.
    rm_uniqnum: an int number, between 1 and data.shape, rm_uniqnum=10,
        if a feature's num of unique values is samller than the unique number,
        remove this feature.
    rm_snratio: a small float number, between 0 and 1, defualt rm_snratio=0.01,
        if a sn's observation of all the variables is smaller than rm_snratio,
        remove this sn, i.e. remove this observation.
        if larger than rm_snratio, keep this sn.
    '''
    import os
    import pickle
    import pandas as pd
    import matplotlib.pyplot as plt
    from .utils import diff, write_iter2txt
    root_path, docname = os.path.split(data_from)
    info_path = os.path.join(root_path, docname + '_Info')
    if not os.path.exists(info_path):
        os.mkdir(info_path)
    # step0: separate the header names and data names.
    d_names = [j for j in data.columns if j.startswith('R3_') and '_R2_' in j]
    d_head = diff(data.columns, d_names)
    len(d_names), len(d_head)
    data.isnull().values.any()  # to check after fill, if there is any nan
    data.shape
    fname = os.path.split(filename)[1]
    # d_names = list(set.difference(set(data.columns.tolist()), set(d_head)))
    txt_name = os.path.join(info_path, fname[:-4] + '_names_variable.txt')
    write_iter2txt(d_names, txt_name)
    txt_name = os.path.join(info_path, fname[:-4] + '_names_header.txt')
    write_iter2txt(d_head, txt_name)
    # d_head = utils.fetch_model_SN(txt_name)
    # step1: remove the columns with dtype of object, remains float, int.
    # step1-1: data_obj_convert to float,
    # step1-2: remove the columns with dtype of object.
    # # the Xdata information only
    dx = data[d_names].dtypes  # the Xdata information only
    obj_names_1 = [i for i in dx.index if dx.at[i] == 'object']
    # df_obj = data[obj_names_1]
    df_new, fs_names = data_obj_convert(data, obj_names_1)
    if len(fs_names) > 0:
        obj_names = diff(obj_names_1, fs_names)
        data[fs_names] = df_new[fs_names]
    else:
        obj_names = obj_names_1
    t_names = diff(d_names, obj_names)
    '''
    len(obj_names), len(fs_names), len(obj_names_1)
    df_new.shape, data.shape
    dt = data[obj_names]; bb = dt[:100]
    # not drop duplicated here
    df = data[['SERIAL_NUMBER'] + t_names].drop_duplicates()
    df.columns[:10]
    # df2 = data[t_dense].drop_duplicates()  # only 2 more removed
    df = df.reset_index(drop=True)
    '''
    txt_name = os.path.join(info_path, fname[:-4] + '_step1_names_obj.txt')
    write_iter2txt(obj_names, txt_name)
    txt_name = os.path.join(info_path, fname[:-4] + '_step1_names_float.txt')
    write_iter2txt(t_names, txt_name)
    # obj_names = list(utils.fetch_model_SN(txt_name))
    # t_names = list(utils.fetch_model_SN(txt_name))
    # len(obj_names) + len(t_names), data.shape[1]
    # step2: remove the columns with nan ratio which is too large >0.2.
    rr = data[t_names].isnull()  # keeps the data names only
    r_nan = rr.sum(axis=0)  # import numpy as np # np.unique(r_nan)
    r_nan = r_nan.sort_values(ascending=False)
    r_ratio = r_nan/data.shape[0]  # r_ratio
    len(set(r_ratio))
    rt_nan = pd.DataFrame(r_ratio, columns=['VALUE'])
    # the data information
    rt_nan_sorted = rt_nan.sort_values(by=['VALUE'], ascending=False)
    rt_name = os.path.join(info_path, fname[:-4] + '_ITEM_NAN_RATIO.csv')
    rt_nan_sorted.to_csv(rt_name)
    nan_names = [j for j in rt_nan.index if rt_nan.loc[j, 'VALUE'] >= rm_nanratio]
    len(nan_names), len(t_names)
    # v_names = [j for j in rt_nan.index if rt_nan.loc[j, 'VALUE'] < rm_nanratio]
    v_names = diff(t_names, nan_names)
    len(v_names), len(nan_names), len(t_names)
    txt_name = os.path.join(info_path, fname[:-4] + '_step2_names_nan20.txt')
    if len(nan_names) > 0:
        write_iter2txt(nan_names, txt_name)
    txt_name = os.path.join(info_path, fname[:-4] + '_step2_names_nanless.txt')
    if len(v_names) > 0:
        write_iter2txt(v_names, txt_name)
    # nan_names = list(utils.fetch_model_SN(txt_name))
    # v_names = list(utils.fetch_model_SN(txt_name))
    # len(v_names), len(nan_names), len(v_names) + len(nan_names), data.shape
    # dx2 = rt_nan[rt_nan.index.isin(v_names)]
    # step3: for each column, remove columns that len(unique values) < 100.
    data3 = data[v_names]   # 'SERIAL_NUMBER' not in v_names
    set(data3.dtypes)
    n_value = {j: len(set(data3[j].dropna())) for j in v_names}
    txt_name = os.path.join(info_path, fname[:-4] + '_num_uniques.txt')
    write_iter2txt(n_value, txt_name)
    nv = pd.DataFrame(
        list(n_value.items()), index=list(n_value.keys()),
        columns=['name', 'n_unique'])
    nv['uniq_ratio'] = [v/data3.shape[0] for k, v in n_value.items()]
    # nv_sorted = nv.sort_values(by=['n_unique'])
    # cu = [nv.loc[i, 'name'] for i in nv.index if nv.loc[i, 'n_unique'] == 1]
    # df_uniq = data[cu]
    # col_sparse = [nv.loc[i, 'name'] for i in nv.index if nv.loc[i, 'uniq_ratio'] < 0.05]
    col_sparse = [nv.loc[i, 'name'] for i in nv.index if nv.loc[i, 'n_unique'] <= rm_uniqnum]
    col_dense = diff(v_names, col_sparse)
    len(col_sparse), len(col_dense)
    # df_sparse = data[col_sparse]
    txt_name = os.path.join(info_path, fname[:-4] + '_step3_names_sparse.txt')
    if len(col_sparse) > 0:
        write_iter2txt(col_sparse, txt_name)
    txt_name = os.path.join(info_path, fname[:-4] + '_step3_names_dense.txt')
    if len(col_dense) > 0:
        write_iter2txt(col_dense, txt_name)
    # col_dense = list(utils.fetch_model_SN(txt_name))
    # len(col_sparse) + len(col_dense), data3.shape
    # step3+: plot and show some figures
    fig_path = os.path.join(root_path, docname + '_Figures')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)  # 创建多级目录
    for j in col_sparse:
        # print(j); break  # len(col_sparse)
        x = data[j].dropna()
        if len(set(x)) > 1:
            fig = plt.figure()
            # the histogram of the data
            n, bins, patches = plt.hist(x, bins=50, density=True, alpha=0.75)
            # plt.hist(x, bins=50, density=True, facecolor='g', alpha=0.75)
            # plt.hist(x, density=False, bins=30)
            plt.ylabel('Probability')
            j2 = rep_illegal_char(j)
            fig_name = str(nv.loc[j, 'n_unique']) + '_' + j2 + '.png'
            fig.savefig(os.path.join(fig_path, fig_name), dpi=200)
            plt.close(fig)
    # step4: to handle the rows with too many nan
    df = data[d_head + col_dense].drop_duplicates()
    df = df.reset_index(drop=True)
    len(col_dense), len(data.columns), df.shape
    # aa = df[:100]
    # df2 = data[col_dense].drop_duplicates()  # only 2 more removed
    rr = df[col_dense].isnull().sum(axis=1)  # each row get a value
    rr = rr.sort_values(ascending=False)  # ww = set(rr)
    rt_row = pd.DataFrame(rr/len(col_dense), columns=['VALUE'])
    rt_row['SERIAL_NUMBER'] = data['SERIAL_NUMBER']
    rt_row = rt_row[['SERIAL_NUMBER', 'VALUE']]
    rt_row_sorted = rt_row.sort_values(by='VALUE', ascending=False)
    rt_name = os.path.join(info_path, fname[:-4] + '_nan_row.csv')
    rt_row_sorted.to_csv(rt_name, index=False)
    # step4-1: get row data with small nan ratio, drop obs with large ratio
    # df1 = df.loc[rt_row['VALUE'] < 0.2, ]  # dangerous!!!
    sn_rm_list = rt_row[rt_row['VALUE'] > rm_snratio]['SERIAL_NUMBER']
    if len(sn_rm_list) > 0:
        df1 = df[~df['SERIAL_NUMBER'].isin(sn_rm_list)]
    else:
        df1 = df
    df1.shape
    df1 = df1.reset_index(drop=True)
    df1 = df1.dropna(axis=1, how='all')
    rr = df1[col_dense].isnull().sum(axis=1)
    rr = rr.sort_values(ascending=False)  #
    set(rr)
    clean_path = os.path.join(root_path, docname + '_Cleaned')
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)  # 创建多级目录
    '''
    filename = os.path.join(clean_path, fname[:-4] + '_CLEANED_head.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filename = os.path.join(clean_path, fname[:-4] + '_CLEANED_head.csv')
    df1.to_csv(filename, index=False)
    '''
    # without d_head, only has 'SERIAL_NUMBER'
    df2 = df1[['SERIAL_NUMBER'] + col_dense]
    filename = os.path.join(clean_path, fname[:-4] + '_CLEANED.pkl')
    with open(filename, 'wb') as handle:
        pickle.dump(df2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filename = os.path.join(clean_path, fname[:-4] + '_CLEANED.csv')
    df2.to_csv(filename, index=False)


def Xclean_whole_path(data_path, docname, ntype):
    '''
    data_path = r'D:/HWtmp2019'
    docname = 'XtableDateFold'
    ntype = 'nor'
    '''
    import os
    import pickle
    import glob
    pre_path = os.path.join(data_path, 'PreData')
    filename = os.path.join(pre_path, 'data_items_' + ntype + '_station.pkl')
    if not os.path.exists(filename):
        return
    with open(filename, 'rb') as handle:
        data_items = pickle.load(handle)
    data_from = os.path.join(data_path, docname)
    for st in data_items:
        print(st)
        flist = glob.glob(data_from + '/' + st + '*' + ntype + '*Xtable.pkl')
        filename = flist[0]  # MUST ONLY ONE
        print(filename)
        # load the data
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        X_data_clean(data_from, filename, data)


def Xclean_whole_path_2_XtableCleaned(data_path, docname):
    import os
    import shutil
    Xclean_whole_path(data_path, docname, ntype='nor')
    Xclean_whole_path(data_path, docname, ntype='exp')
    # create a backup directory
    SOURCE = os.path.join(data_path, docname + '_Cleaned')
    BACKUP = os.path.join(data_path, 'XtableCleaned')
    shutil.copytree(SOURCE, BACKUP)
    print(os.listdir(BACKUP))


class CleanXdata():

    def __init__(self, data_path, docname, rm_nanratio=0.2, rm_uniqnum=10,
                 rm_snratio=0.01):
        """
        Parameters
        rm_nanratio: a float number, between 0 and 1, rm_nanratio=0.2,
            if a feature's nan ratio is bigger than rm_nanratio,
            remove this feature.
        rm_uniqnum: an int number, between 1 and data.shape, rm_uniqnum=10,
            if a feature's num of unique values is samller than the unique
            number,remove this feature.
        rm_snratio: a small float number, between 0 and 1, defualt
            rm_snratio=0.01, if a sn's observation of all the variables is
            smaller than rm_snratio, remove this sn, i.e. remove this
            observation. If larger than rm_snratio, keep this sn.
        # data_path = r'D:/HWtmp2019'
        # docname = 'XtableDateFold'
        """
        self.__data_path = data_path
        self.__docname = docname
        self.rm_nanratio = rm_nanratio
        self.rm_uniqnum = rm_uniqnum
        self.rm_snratio = rm_snratio

    def _Xclean_whole_path(self, ntype):
        # ntype = 'nor' or 'exp'
        import os
        import pickle
        import glob
        data_path = self.__data_path
        docname = self.__docname
        rm_nanratio = self.rm_nanratio
        rm_uniqnum = self.rm_uniqnum
        rm_snratio = self.rm_snratio
        pre_path = os.path.join(data_path, 'PreData')
        filename = os.path.join(pre_path, 'data_items_' + ntype + '_station.pkl')
        if not os.path.exists(filename):
            return
        with open(filename, 'rb') as handle:
            data_items = pickle.load(handle)
        data_from = os.path.join(data_path, docname)
        for st in data_items:
            print(st)
            flist = glob.glob(data_from + '/' + st + '*' + ntype + '*Xtable.pkl')
            filename = flist[0]  # MUST ONLY ONE
            print(filename)
            # load the data
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            X_data_clean(data_from, filename, data, rm_nanratio, rm_uniqnum,
                         rm_snratio)

    def _Xclean_whole_path_2_XtableCleaned(self):
        import os
        import shutil
        self._Xclean_whole_path(ntype='nor')
        self._Xclean_whole_path(ntype='exp')
        # create a backup directory
        data_path = self.__data_path
        docname = self.__docname
        SOURCE = os.path.join(data_path, docname + '_Cleaned')
        BACKUP = os.path.join(data_path, 'XtableCleaned')
        shutil.copytree(SOURCE, BACKUP)
        print(os.listdir(BACKUP))

    def xclean(self):
        self._Xclean_whole_path_2_XtableCleaned()






