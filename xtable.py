# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:58:41 2019

@author: HaiyanJiang
@email : jianghaiyan.cn@gmail.com


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

"""


def extract_data_info(from_path, data_path, sel_names, s_dtype, keep_names,
                      rename_dict):
    import os
    import glob
    import time
    from .utils import log_read_file, ReadBigCsv
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    log_path = os.path.join(data_path, 'LOGS')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cur_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_name = os.path.join(log_path, 'log_slim_@' + cur_date[:10] + '.txt')
    rbc = ReadBigCsv(sel_names, s_dtype, keep_names, rename_dict, data_path)
    flist = glob.glob(from_path + '/*.csv')  # len(flist)
    for input_file in flist:  # input_file = flist[0]
        print(input_file)

        @log_read_file(log_name, input_file)
        def log_extract_data_info():
            rbc.extract_bigcsv_item_info(input_file, sep=',')
        log_extract_data_info()


def keepday_cuthour(x):
    # x is string, like x = '2019-03-02 13:03:58.0'
    # x1 = '2019-03-02'; x2 = '2019-03-02'; x1 == x2
    if len(x) >= 10:
        return x[:10]
    else:
        return x


def read_pkl_save_Xtable(data_items, filename, to_path):
    # directly extract filename and save to Xtable with separate stations
    import os
    import pickle
    from .utils import chunk_2_Xtable
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    data.columns, data.shape  # aa = data[:100]
    data['TEST_DATE'] = [x[:10] if len(x) >= 10 else x for x in data['TEST_DATE']]
    st_list = [s for s in set(data['STATION_NAME']) if isinstance(s, str)]
    fname = os.path.split(filename)[1]
    print(fname)
    for st in st_list:  # st = st_list[0]
        station_name = st.replace('/', '')  # need some modifies
        print(station_name)
        out_path = os.path.join(to_path, station_name)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        # st = 'B/C'
        df = data[data['STATION_NAME'] == st]
        df = df.reset_index(drop=True).drop(['STATION_NAME'], axis=1)
        df.columns
        df.shape
        item_ID = data_items[station_name]
        # dx1, dx2 = chunk_2_Xtable(df[:2000], item_ID)
        dx1, dx2 = chunk_2_Xtable(df, item_ID)
        # len(set(df['SERIAL_NUMBER']))
        out_file = os.path.join(out_path, fname[:-4] + '_size.pkl')
        with open(out_file, "wb") as handle:
            pickle.dump(dx1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        filename = os.path.join(out_path, fname[:-4] + '_Xtable.pkl')
        with open(filename, 'wb') as handle:  # protocol = 4
            pickle.dump(dx2, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(data_path, ntype='nor'):
    '''
    data_path = r'D:/HWtmp2019'
    ntype = 'nor'
    '''
    import time
    import os
    import glob
    import pickle
    from .utils import log_read_file
    to_path = os.path.join(data_path, 'XtableStation')
    if not os.path.exists(to_path):
        os.makedirs(to_path)
    pre_path = os.path.join(data_path, 'PreData')
    filename = os.path.join(pre_path, 'data_items_' + ntype + '_station.pkl')
    if not os.path.exists(filename):
        return
    with open(filename, 'rb') as handle:
        data_items = pickle.load(handle)
    filename = os.path.join(pre_path, 'data_sn_' + ntype + '_station.pkl')
    with open(filename, 'rb') as handle:
        data_sn = pickle.load(handle)
    len(data_sn)
    slim_path = os.path.join(data_path, 'SlimData')
    flist = glob.glob(slim_path + '/*' + ntype + '*.pkl')  # len(flist)
    if len(flist) == 0:
        return
    log_path = os.path.join(data_path, 'LOGS')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cur_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_name = os.path.join(log_path, 'log_readpkl_@' + cur_date[:10] + '.txt')
    for filename in flist:  # filename = flist[0]
        print(filename)

        @log_read_file(log_name, filename)
        def log_read_pkl():
            read_pkl_save_Xtable(data_items, filename, to_path)
        log_read_pkl()


def combine_station(data_path, ntype='nor'):
    # read the pickles and append each pickle to a desired dataframe.
    # data_path = r'D:/HWtmp2019'; ntype = 'nor'
    '''
    pre_path = os.path.join(data_path, 'PreData')
    filename = os.path.join(pre_path, 'data_items_' + ntype + '_station.pkl')
    with open(filename, 'rb') as handle:
        data_items = pickle.load(handle)
    '''
    import os
    import glob
    import pickle
    import pandas as pd
    data_to = os.path.join(data_path, 'XtableTMP')
    if not os.path.exists(data_to):
        os.makedirs(data_to)
    st_path = os.path.join(data_path, 'XtableStation')
    st_list = os.listdir(st_path)
    for st in st_list:  # st = st_list[0]
        pickle_path = os.path.join(st_path, st)
        # where the pickle files are.
        flist = glob.glob(pickle_path + '/*' + ntype + '*_Xtable.pkl')
        if len(flist) == 0:
            continue
        # pd.read_pickle(filename); pickle.load(handle); len(flist)
        # add it to a list; then pd.concat(thelist)
        # df = df.append(chunk, ignore_index=True) is very slooowly.
        chunks = []
        for filename in flist:
            with open(filename, 'rb') as handle:
                chunk = pickle.load(handle)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        df = df.drop_duplicates().reset_index(drop=True)
        zname = st + '_' + ntype + '_Xtable'
        # add the station name to the first of the dataname
        filename = os.path.join(data_to, zname + '.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        filename = os.path.join(data_to, zname + '.csv')
        df.to_csv(filename, index=False)


def vec_mean_most(x):
    '''
    as for numerical data, if there is not only one data, get the mean;
    as for character data, if there is not only one data, get the mode;
    if the mixture of numer and char, get the mean of the numerical part.
    for data like '12.2', which can be viewed as numerical data.
    '''
    import pandas as pd
    from collections import Counter
    s = x.dropna(axis=0, how='any')
    # 单个的字符 or 数字
    if len(s) == 1:
        s1 = pd.to_numeric(s, errors='ignore')  # downcast=None
        return s1.tolist()[0]
    # 混杂的话，取数值型的 mean # ['a', '1', '2'] to [NaN, 1.0, 2.0]
    z1 = pd.to_numeric(s, errors='coerce')
    z2 = z1.dropna()
    if len(z2) > 0:
        return z2.mean()
    else:  # 可能全是字符型
        z = pd.to_numeric(s, errors='ignore')  # downcast=None
        # 全部空字符nan的话，直接转化成 nan, 因此不需要 z1 = z.dropna()
        if 'object' == z.dtype:
            v = Counter(z).most_common(1)[0][0]
            return v
        else:
            return z.mean()  # the mean


def vec_max_most(x):
    '''
    as for numerical data, if there is not only one data, get the max;
    as for character data, if there is not only one data, get the mode;
    if the mixture of numer and char, get the max of the numerical part.
    for data like '12.2', which can be viewed as numerical data.
    '''
    import pandas as pd
    from collections import Counter
    s = x.dropna(axis=0, how='any')
    # 单个的字符 or 数字
    if len(s) == 1:
        s1 = pd.to_numeric(s, errors='ignore')  # downcast=None
        return s1.tolist()[0]
    # 混杂的话，取数值型的 mean # ['a', '1', '2'] to [NaN, 1.0, 2.0]
    z1 = pd.to_numeric(s, errors='coerce')
    z2 = z1.dropna()
    if len(z2) > 0:
        return z2.max()
    else:  # 可能全是字符型
        z = pd.to_numeric(s, errors='ignore')  # downcast=None
        # 全部空字符nan的话，直接转化成 nan, 因此不需要 z1 = z.dropna()
        if 'object' == z.dtype:
            v = Counter(z).most_common(1)[0][0]
            return v
        else:
            return z.max()  # the max


def xtable_fold_column_keep_meanmax(data, b_names, column, keep_fun):
    # data is already numerical data
    # fold R3_RESULT, and finally drop column from data. [version 1]
    # fold R3_RESULT, finally donot drop column from data, keep max value. [v2]
    # #### when folding R3_RESULT, all the three must be the same.
    # column = ['R3_RESULT']
    # keep_fun = vec_mean_most
    # dg = data.groupby(['SERIAL_NUMBER']).size()
    import pandas as pd
    from .utils import diff
    data = data.drop_duplicates().reset_index(drop=True)
    sel_col = diff(data.columns, column)  # len(sel_col)
    dg = data.groupby(b_names).size()
    dx = dg.reset_index().rename(columns={0: 'count'})
    # 将 count = 1 的直接拿出来
    ds = dx[dx['count'] > 1]  # special
    if len(ds) == 0:
        return None, data
    '''
    sn_dup = list(set(ds['SERIAL_NUMBER']))  # 仅仅使用 sn 是分不开的
    sn = '5T3JPM163S012908'
    sn = '5T3JPM163M005014'
    aa = data[data['SERIAL_NUMBER'] == sn]
    len(aa), len(sn_dup)  # sn in sn_dup
    bb = df_dup[df_dup['SERIAL_NUMBER'] == sn]
    cc = df_uni[df_uni['SERIAL_NUMBER'] == sn]
    '''
    # this can be totally seperate data into df_uni and df_dup
    ds = dx[dx['count'] > 1]  # special,
    dp = ds.reset_index(drop=True).drop(columns='count')
    df_dup = pd.merge(data, dp, on=b_names, how='inner')
    # aa = df_dup[:10]
    du = dx[dx['count'] == 1]  # special, unique ones
    df_uni = pd.merge(data, du.drop(columns='count'), on=b_names, how='inner')
    df_chunks = []
    for i in range(len(dp)):  # i = 0
        # print(i); break
        # x = dp.loc[[i]]  # dataframe with one row
        xmg = pd.merge(df_dup, dp.loc[[i]], on=b_names, how='inner')
        xt = xmg.dropna(axis=1, how='all')  # axis = 1 is for columns
        xt = xmg[sel_col].dropna(axis=1, how='all')  # sel_col
        # xt2 = xt.apply(vec_mean_most, axis=0)  # 遍历行
        xt2 = xt.apply(keep_fun, axis=0)  # 遍历行
        # dp.loc[i, 'n_feat'] = len(xt2) - 1  # SERIAL_NUMBER
        xx = dict(xt2)
        for j in column:
            if j == 'TEST_NUM':
                xx[j] = len(set(xmg[j]))  # test_num as the unique len
            else:
                xx[j] = xmg[j].max()  # keep the max of column j.
        xtmp = pd.DataFrame(columns=data.columns.values)
        xtmp = xtmp.append(xx, ignore_index=True)
        # # zz = xtmp[cname_list].dropna(axis=1, how='all')
        df_chunks.append(xtmp)
    df_tmp = pd.concat(df_chunks, ignore_index=True)
    if 'TEST_NUM' in column:
        df_uni['TEST_NUM'] = 1
    df = pd.concat([df_uni, df_tmp], ignore_index=True)
    # ############# df = df.drop(axis=1, columns=column)  # no drop column
    # df = df.apply(pd.to_numeric, errors='ignore')  # bb =df[:100]
    # ### finnally return ds not dp
    return ds, df


def xtable_fold_column_keep_max(data, b_names, column):
    return xtable_fold_column_keep_meanmax(
            data, b_names, column, keep_fun=vec_max_most)


def xtable_fold_column_keep_mean(data, b_names, column):
    return xtable_fold_column_keep_meanmax(
            data, b_names, column, keep_fun=vec_mean_most)


def xtable_fold_column(data_from, kcol, column, gb_names, ntype='nor'):
    '''
    pre_path = os.path.join(root_path, 'PreData')
    filename = os.path.join(pre_path, 'data_items_' + ntype + '_station.pkl')
    with open(filename, 'rb') as handle:
        data_items = pickle.load(handle)
    '''
    import os
    import glob
    import pickle
    import pandas as pd
    from .utils import diff
    # root_path = r'D:/HWtmp2019'; ntype = 'nor'
    root_path = os.path.split(data_from)[0]
    to_path = os.path.join(root_path, 'Xtable' + kcol + 'Fold')
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    flist = glob.glob(data_from + '/*' + ntype + '*_Xtable.pkl')
    if len(flist) == 0:
        return
    for filename in flist:  # filename = fname_list[0]
        fname = os.path.split(filename)[1][:-4]
        print(fname)
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        data = data.apply(pd.to_numeric, errors='ignore')  # bb = data[:100]
        # data.shape; aa_col2 = data.columns[:20].tolist()
        if kcol == 'R3':
            keep_path = os.path.join(root_path, 'Xtable' + kcol + 'Fail')
            if not os.path.exists(keep_path):
                os.mkdir(keep_path)
            tmp = data[data['R3_RESULT'] == 1]
            sn_list = list(set(tmp['SERIAL_NUMBER']))  # len(sn_list)
            if len(sn_list) > 0:
                df1 = data[data['SERIAL_NUMBER'].isin(sn_list)]
                df1.shape[0] / data.shape[0]
                filename_1 = os.path.join(keep_path, fname + '_R3FAIL.pkl')
                with open(filename_1, 'wb') as handle:
                    pickle.dump(df1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        '''
        # fold R3_RESULT
        st = fname.split('_')[0]  # the fist saves the station name
        h_names = diff(data.columns, data_items[st])  # len(h_names)
        gb_names = diff(h_names, column + ['ATE_ID', 'TPS_ID'])
        data.columns[:5]
        aa = data[:100]
        ds, df = xtable_fold_column_keep_mean(aa, b_names, column)
        '''
        b_names = diff(gb_names, column)
        # column = ['R3_RESULT']  # exclude '*_ID' something
        ds, df = xtable_fold_column_keep_mean(data, b_names, column)
        df = df.dropna(axis=1, how='all')  # A MUST
        # 要dropna(axis=1)是去看是否有变量只在这些特殊的 sn 上有取值
        # ################ to save ds which counts the number of features
        if ds is not None:
            # print('ds is NOT NONE!')
            filename = os.path.join(to_path, fname + '_' + kcol + '_STAT.csv')
            ds.to_csv(filename, index=False)
        # ###################### to path ###################################
        filename = os.path.join(to_path, fname + '.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        filename = os.path.join(to_path, fname + '.csv')
        df.to_csv(filename, index=False)


def xtable_folded(data_path, gb_names, fold_cols, sub_docs, ntype):
    # ntype = 'exp'; ntype = 'nor'
    import os
    import time
    from .utils import diff, log_read_file
    if not gb_names:
        print('No value of gb_names.')
        return
    if 'SERIAL_NUMBER' not in gb_names:
        ValueError('Invalid gb_names, which is a list and must include '
                   'SERIAL_NUMBER. Check the list of available parameters')
        print('Invalid gb_names, which is a list and must include '
              'SERIAL_NUMBER. Check the list of available parameters.')
        return
    if len(diff(gb_names, ['SERIAL_NUMBER'])) == 0:
        print('No value of gb_names except SERIAL_NUMBER.')
        return
    log_path = os.path.join(data_path, 'LOGS')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    cur_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_name = os.path.join(log_path, 'log_xtable_@' + cur_date[:10] + '.txt')
    for kcol, column in fold_cols.items():
        # kcol, column = 'R3', ['R3_RESULT']
        # kcol, column = 'Date', ['TEST_DATE']
        # print(kcol); break
        print(kcol)
        data_from = os.path.join(data_path, sub_docs[kcol])

        @log_read_file(log_name, data_from)
        def log_xfold():
            xtable_fold_column(data_from, kcol, column, gb_names, ntype)
        log_xfold()


class WIXtable():

    def __init__(self, from_path, data_path,
                 sel_names, s_dtype, keep_names, rename_dict,
                 gb_names, fold_cols, sub_docs):
        """
        Parameters
        ----------
        from_path: raw data path, must be a full path name, not absolute path.
        data_path: the data path, full path, where we want to save the data.
        sel_names: list,
            selected selected column names from the raw data in 'from_path',
            where the following 5 must be inclued 'sn', 'r2testitemname',
            'r3subtestitemname', 'r3resultdesc', 'wsname'.
        s_dtype: dict,
            specifying the format of data, must be a corresponding dictionary
            to sel_names and we can skip 'r3resultdesc'.
        rename_dict: dict,
            replace the old column name by the new one, must be a corresponding
            dictionary to sel_names.
        keep_names: list,
            must have at least ['sn', 'r2testitemname', 'r3subtestitemname']
        NOTES:
            the sel_names, s_dtype, rename_dict must include the same name,
            except that there is no 'r3resultdesc' in stype, because
            'r3resultdesc' might be 'str' or 'float', we donot know exactly
            what type it is.
        gb_names: list, groupby names,
            ['SERIAL_NUMBER', 'TEST_DATE', 'UUT_NAME', 'R3_RESULT'],
        fold_cols: dict, keyword and a one length list.
        sub_docs: dict,
            data used for folding and data saved file for each document
        """
        self.from_path = from_path
        self.data_path = data_path
        self.sel_names = sel_names
        self.s_dtype = s_dtype
        self.keep_names = keep_names
        self.rename_dict = rename_dict
        self.gb_names = gb_names
        self.fold_cols = fold_cols
        self.sub_docs = sub_docs

    def _extract_data_info(self):
        import os
        import glob
        import time
        from .utils import log_read_file, ReadBigCsv
        from_path = self.from_path
        data_path = self.data_path
        sel_names = self.sel_names
        s_dtype = self.s_dtype
        keep_names = self.keep_names
        rename_dict = self.rename_dict
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        log_path = os.path.join(data_path, 'LOGS')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        cur_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_name = os.path.join(log_path, 'log_slim_@' + cur_date[:10] + '.txt')
        rbc = ReadBigCsv(
                sel_names, s_dtype, keep_names, rename_dict, data_path)
        flist = glob.glob(from_path + '/*.csv')  # len(flist)
        for input_file in flist:  # input_file = flist[4]
            print(input_file)

            @log_read_file(log_name, input_file)
            def log_extract_data_info():
                rbc.extract_bigcsv_item_info(input_file, sep=',')
            log_extract_data_info()

    def _slim_bigcsv(self):
        # ntype='nor' or ntype='exp'
        from .utils import combine_itemnames_sn, get_station_names
        data_path = self.data_path
        self._extract_data_info()
        combine_itemnames_sn(data_path, ntype='nor')
        combine_itemnames_sn(data_path, ntype='exp')
        get_station_names(data_path, ntype='nor')
        get_station_names(data_path, ntype='exp')

    def _readpkl_slimdata(self):
        data_path = self.data_path
        read_pkl(data_path, ntype='nor')
        read_pkl(data_path, ntype='exp')
        combine_station(data_path, ntype='nor')
        combine_station(data_path, ntype='exp')

    def _xtable_folded(self):
        # ntype = 'exp'; ntype = 'nor'
        data_path = self.data_path
        gb_names = self.gb_names
        fold_cols = self.fold_cols
        sub_docs = self.sub_docs
        xtable_folded(data_path, gb_names, fold_cols, sub_docs, ntype='nor')
        xtable_folded(data_path, gb_names, fold_cols, sub_docs, ntype='exp')

    def wixtable(self):
        self._slim_bigcsv()
        self._readpkl_slimdata()
        self._xtable_folded()

