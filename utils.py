# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:19:29 2019

@author: HaiyanJiang
@email : jianghaiyan.cn@gmail.com

"""


def diff(first, second):
    '''
    Use set if you don't care about items order or repetition.
    Use list comprehensions if you do
    '''
    # list(set.difference(set(first), set(second)))
    second = set(second)
    return [item for item in first if item not in second]


def read_bigcsv(filename, **kw):
    import pandas as pd
    kw.update(dict(error_bad_lines=False, warn_bad_lines=False))
    chunkSize = 1000000
    chunks = []
    with open(filename) as rf:
        # reader = pd.read_csv(rf, **kw, dtype=s_dtype, iterator=True)
        reader = pd.read_csv(rf, iterator=True, **kw)
        go = True
        while go:
            try:
                chunk = reader.get_chunk(chunkSize)  # iterator
                chunks.append(chunk)
            except StopIteration:
                # except Exception as e:
                # type(e) # <class 'StopIteration'>  # print(e) nothing
                # print(type(e), "Iteration is stopped.")
                go = False
    df = pd.concat(chunks, axis=0, join='outer', ignore_index=True)
    # default axis=0, join='outer'
    return df


def chunk_bigfile(filename, **kw):
    # import sys
    # sys.getdefaultencoding()
    import pandas as pd
    kw.update(dict(error_bad_lines=False, warn_bad_lines=False))
    chunkSize = 100000  # 10
    chunks = []
    # rf = open(filename)
    # rf = open(filename, 'r', encoding='GB18030')
    with open(filename, 'r', encoding='GB18030') as rf:
        # reader = pd.read_csv(rf, **kw, dtype=s_dtype, iterator=True)
        reader = pd.read_csv(rf, iterator=True, **kw)
        go = True
        while go:
            try:
                chunk = reader.get_chunk(chunkSize)  # iterator
                chunks.append(chunk)
            except StopIteration:
                # except Exception as e:  # type(e), StopIteration
                # type(e) # <class 'StopIteration'>
                # print(type(e), "Iteration is stopped.")
                # except Exception as e:  # type(e), StopIteration
                # print(type(e), "Iteration is stopped.")
                go = False
    return chunks


def read_csv_top_k(filename, k=50, **kw):
    import pandas as pd
    chunkSize = k
    with open(filename) as rf:
        reader = pd.read_csv(rf, **kw, iterator=True)
        df = reader.get_chunk(chunkSize)
    k2 = min(df.shape[0], k)
    res = df[:k2]
    return res


def write_iter2txt(iter_data, txt_name):
    if isinstance(iter_data, dict):
        with open(txt_name.encode('utf8'), 'w') as f:
            for item in iter_data:
                f.write("%s,%s,\n" % (item, iter_data[item]))
    elif isinstance(iter_data, (set, list)):
        with open(txt_name.encode('utf8'), 'w') as f:
            for item in iter_data:
                f.write("%s,\n" % item)


def fetch_SN(tname):
    # return a set
    import os
    import pandas as pd
    if not os.path.isfile(tname):
        return None
    try:
        df_sn = pd.read_csv(tname, sep=',', header=None, encoding='utf-8')
    except Exception as e:
        print(e)
        # df_sn = pd.read_csv(tname, sep=',', header=None, encoding='gbk')
        df_sn = pd.read_csv(
            tname, sep=',', header=None, encoding='gbk', engine='python')
    df_sn = df_sn.dropna(axis=1, how='all')  # axis=1, columns
    df_sn = df_sn.dropna(axis=0, how='any')  # axis=0, row
    sn = set(df_sn[0])  # get the fisrt column
    return sn


def log_read_file(log_name, file_name):
    import time
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*arg, **kw):
            ck0 = time.process_time()
            ctime_0 = time.ctime()
            try:
                fn = func(*arg, **kw)
                ck1 = time.process_time()
                ctime_1 = time.ctime()
                # 防止多个文件同时执行时, write into the log冲突. 'a' append.
                with open(log_name, 'a') as f:  # 防止多个文件同时执行时log冲突.
                    f.write('%s() begins on %s\n' % (func.__name__, ctime_0))
                    f.write('FILENAME: %s \n' % file_name)
                    f.write("SUCCEED!\n")
                    f.write('time used: %f \n' % (ck1-ck0))
                    f.write('%s() ends on %s\n\n\n' % (func.__name__, ctime_1))
                return fn
            except Exception as e:
                ck1 = time.process_time()
                ctime_1 = time.ctime()
                with open(log_name, 'a') as f:
                    f.write('%s() begins on %s\n' % (func.__name__, ctime_0))
                    f.write('FILENAME: %s \n' % file_name)
                    f.write("FAIL!\t %s: %s\n" % (type(e), e))
                    f.write('time used: %f \n' % (ck1-ck0))
                    f.write('%s() ends on %s\n\n\n' % (func.__name__, ctime_1))
        return wrapper
    return decorator


def log_func(log_name):
    import time
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*arg, **kw):
            ck0 = time.process_time()
            ctime_0 = time.ctime()
            try:
                fn = func(*arg, **kw)
                ck1 = time.process_time()
                ctime_1 = time.ctime()
                # 防止多个文件同时执行时, write into the log冲突. 'a' append.
                with open(log_name, 'a') as f:  # 防止多个文件同时执行时log冲突.
                    f.write('%s() begins on %s\n' % (func.__name__, ctime_0))
                    f.write('SUCCEED!\n')
                    f.write('time used: %f \n' % (ck1-ck0))
                    f.write('%s() ends on %s\n\n\n' % (func.__name__, ctime_1))
                return fn
            except Exception as e:
                ck1 = time.process_time()
                ctime_1 = time.ctime()
                with open(log_name, 'a') as f:
                    f.write('%s() begins on %s\n' % (func.__name__, ctime_0))
                    f.write('FAIL!\t %s: %s\n' % (type(e), e))
                    f.write('time used: %f \n' % (ck1-ck0))
                    f.write('%s() ends on %s\n\n\n' % (func.__name__, ctime_1))
        return wrapper
    return decorator


def mean_most(df, column):
    '''
    as for numerical data, if there is not only one data, get the mean;
    as for character data, if there is not only one data, get the mode;
    if the mixture of numer and char, get the mean of the numerical part.
    for data like '12.2', which can be viewed as numerical data.
    column can and only can be one of the column, char
    # column = 'RESULT_DESC'
    '''
    import pandas as pd
    from collections import Counter
    s = df[column].dropna(axis=0, how='any')
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
            return z.mean()


def xrow_item(df, item_ID):
    import pandas as pd
    col_r3 = 'R3_FT_TEST_ITEM_NAME'
    col_r2 = 'R2_FT_TEST_ITEM_NAME'
    col_value = 'R3_value'
    key = list('R3_' + df[col_r3] + '_R2_' + df[col_r2])
    x = dict(zip(key, df[col_value]))
    xrow = pd.DataFrame(columns=item_ID)
    xrow = xrow.append(x, ignore_index=True)
    return xrow


def chunk_2_Xtable(df, item_ID):
    # 这里的 g_names 比 gb_names 多的是 'TEST_DATE', 'TEST_NUM', 'R3_RESULT',
    # g_names = [j for j in df.columns if j != 'RESULT_DESC']
    '''
    g_names = [
        'SERIAL_NUMBER', 'TEST_DATE', 'TEST_NUM', 'R3_RESULT',
        'R2_FT_TEST_ITEM_NAME', 'R3_FT_TEST_ITEM_NAME']
    '''
    g_names = diff(df.columns, ['RESULT_DESC'])  # len(g_names)
    dg = df.groupby(g_names).apply(mean_most, 'RESULT_DESC')
    dx = dg.reset_index().rename(columns={0: 'R3_value'})
    # b_names = ['SERIAL_NUMBER', 'TEST_DATE', 'TEST_NUM', 'R3_RESULT']
    exp_col = ['R2_FT_TEST_ITEM_NAME', 'R3_FT_TEST_ITEM_NAME', 'RESULT_DESC']
    b_names = diff(df.columns, exp_col)  # len(b_names)
    dg1 = dx.groupby(b_names).size()
    dx1 = dg1.reset_index().rename(columns={0: 'size'})
    dg2 = dx.groupby(b_names).apply(xrow_item, item_ID)  # len(item_ID)
    dx2 = dg2.reset_index()
    add_cols = diff(diff(dx2.columns, dg2.columns), b_names)
    dx2 = dx2.drop(axis=1, columns=add_cols)
    # dx1 is the summary and dx2 is the xtable data
    return dx1, dx2


def extract_item_info(df, dirname, filename, station_name):
    import os
    import pickle
    # dirname, zname are used for saving data
    fname = os.path.split(filename)[1]
    st = station_name.replace('/', '')
    zname = fname[:-4] + '_' + st
    i_name = os.path.join(dirname, zname + '_R3_ITEM_NAME.txt')
    # r3_ID = np.unique(df['FT_TEST_ITEM_NAME']).tolist()
    item_R3_ID = list(set(df['R3_FT_TEST_ITEM_NAME']))
    len(item_R3_ID)
    write_iter2txt(item_R3_ID, i_name)
    # ###################### save R3R2_ITEM_NAME #################
    # df.columns
    gb_names = [
        'SERIAL_NUMBER', 'R2_FT_TEST_ITEM_NAME', 'R3_FT_TEST_ITEM_NAME']
    dt = df.groupby(gb_names).size().reset_index()
    # dt.columns
    dt = dt.rename(columns={0: 'count'})
    dt = dt.drop(axis=1, columns=['SERIAL_NUMBER', 'count'])
    dt = dt.drop_duplicates(keep='first')
    # # encoding='utf_8_sig'  # 解决中文乱码问题
    r3, r2 = dt['R3_FT_TEST_ITEM_NAME'], dt['R2_FT_TEST_ITEM_NAME']
    item_names = list('R3_' + r3 + '_R2_' + r2)
    len(dt), len(item_names)
    r_name = os.path.join(dirname, zname + '_R3R2_ITEM_NAME.txt')
    write_iter2txt(item_names, r_name)
    # save dt as
    dt['R3R2_ITEM_NAME_JOINT'] = item_names
    it_name = os.path.join(dirname, zname + '_R3R2_ITEM_NAME_JOINT.csv')
    dt.to_csv(it_name, index=False, encoding='utf_8_sig')  # 解决中文乱码问题
    # #######################
    # to save the serial number
    sn_list = list(set(df['SERIAL_NUMBER']))  # len(sn_list)
    sn_name = os.path.join(dirname, zname + '_SERIAL_NUMBER.txt')
    write_iter2txt(sn_list, sn_name)
    # to save the summary txt
    sm = dict(nshape_raw=df.shape, n_item_name=dt.shape[0])
    cnames = [
        'SERIAL_NUMBER', 'TEST_DATE',
        'R2_FT_TEST_ITEM_NAME', 'R3_FT_TEST_ITEM_NAME', 'R3_RESULT']
    for j in cnames:
        sm['n_' + j] = len(set(df[j]))
    sm_name = os.path.join(dirname, zname + '_SUMMARY.txt')
    write_iter2txt(sm, sm_name)
    # aa = df[:10]
    savename = os.path.join(dirname, zname + '_SELCOL.pkl')
    with open(savename, 'wb') as handle:  # protocol=4
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_itemnames_sn(data_path, ntype='nor'):
    '''
    data_path = r'D:/HWtmp2019'
    ntype='exp'; ntype='nor'
    '''
    import os
    import glob
    import pandas as pd
    import pickle
    # TO combine the itemnames and sn within the same station name
    # the following codes are to get the station names
    slim_path = os.path.join(data_path, 'SlimData')
    flist = glob.glob(slim_path + '/*' + ntype + '*.pkl')
    if len(flist) == 0:
        return
    st_names = set()  # station names
    for filename in flist:
        print(filename)
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        st_names = st_names | set(data['STATION_NAME'])
    st_list = [s.replace('/', '') for s in st_names if isinstance(s, str)]
    st_list = sorted(st_list)
    len(st_names), len(st_list), len(set(st_list))
    # TO get the info data
    info_path = os.path.join(data_path, 'INFO')
    data_items, data_sn = {}, {}
    for s in st_list:  # s = st_list[0]
        print(s)
        # read the data items
        it_list = glob.glob(
            info_path + '/*' + ntype + '*' + s + '_R3R2_ITEM_NAME.txt')
        st_set = set()
        for tname in it_list:  # tname = it_list[0]
            # df_sn = pd.read_csv(tname, header=None, encoding='gbk', engine='python')
            # df_sn = pd.read_csv(tname, header=None, encoding='utf_8_sig')
            try:
                df_sn = pd.read_csv(tname, header=None, encoding='utf_8_sig')
            except Exception as e:
                print(e)
                df_sn = pd.read_csv(tname, sep=',', header=None, encoding='gbk', engine='python')
            df_sn = df_sn.dropna(axis=1, how='all')  # axis=1, columns
            df_sn = df_sn.dropna(axis=0, how='any')  # axis=0, row
            # sn = fetch_SN(tname)
            st_set = st_set | set(df_sn[0])  # get the fisrt column
            len(st_set), len(set(df_sn[0]))
        data_items[s] = list(st_set)
        # read the corresponding SN
        sn_list = glob.glob(
            info_path + '/*' + ntype + '*' + s + '_SERIAL_NUMBER.txt')
        st_set = set()
        for tname in sn_list:  # tname = sn_list[0]
            # df_sn = pd.read_csv(tname, sep=',', header=None)
            # df_sn = pd.read_csv(tname, sep=',', header=None, encoding='gbk', engine='python')
            try:
                df_sn = pd.read_csv(tname, header=None, encoding='utf_8_sig')
            except Exception as e:
                print(e)
                df_sn = pd.read_csv(tname, sep=',', header=None, encoding='gbk', engine='python')
            df_sn = df_sn.dropna(axis=1, how='all')  # axis=1, columns
            df_sn = df_sn.dropna(axis=0, how='any')  # axis=0, row
            st_set = st_set | set(df_sn[0])  # get the fisrt column
            len(st_set), len(set(df_sn[0]))
        data_sn[s] = list(st_set)
    # data saving path
    pre_path = os.path.join(data_path, 'PreData')
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)
    # saving the data_items
    savename = os.path.join(pre_path, 'data_items_' + ntype + '_station.pkl')
    with open(savename, 'wb') as handle:  # protocol=4
        pickle.dump(data_items, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # saving the data_sn
    savename = os.path.join(pre_path, 'data_sn_' + ntype + '_station.pkl')
    with open(savename, 'wb') as handle:  # protocol=4
        pickle.dump(data_sn, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # summary of the variable names along all the stations
    txt_name = os.path.join(pre_path, 'SUMMARY_' + ntype + '_sn_items.txt')
    sm = {}
    for s in sorted(st_list):
        sm[s] = [len(data_sn[s]), len(data_items[s])]
    write_iter2txt(sm, txt_name)
    txt_name = os.path.join(pre_path, 'SUMMARY_' + ntype + '_sn_items.csv')
    df_sm = pd.DataFrame.from_dict(
        sm, orient='index', columns=['n_sn', 'n_item'])
    df_sm = df_sm.reset_index()
    df_sm.rename(columns={'index': 'station_name'}, inplace=True)
    df_sm.to_csv(txt_name, index=False, encoding='utf_8_sig')  # 解决中文乱码问题


def get_station_names(data_path, ntype='nor'):
    # data_from = r'D:/HWtmp2019'
    # ntype='exp'
    import glob
    import os
    import pickle
    data_from = os.path.join(data_path, 'SlimData')
    flist = glob.glob(data_from + '/*' + ntype + '*.pkl')
    if len(flist) == 0:
        return
    st_names = set()
    for filename in flist:  # filename = flist[0]
        print(filename)
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        st_names = st_names | set(data['STATION_NAME'])
    # st_list = [s.replace('/', '') for s in st_names if isinstance(s, str)]
    st_names = list(st_names)
    data_to = os.path.join(data_path, 'PreData')
    if not os.path.exists(data_to):
        os.makedirs(data_to)
    txt_name = os.path.join(data_to, ntype + '_STATION_NAMES.txt')
    if len(st_names) > 0:
        write_iter2txt(st_names, txt_name)


class ReadBigCsv():

    def __init__(self, sel_names, s_dtype, keep_names, rename_dict, to_path):
        self.sel_names = sel_names
        self.s_dtype = s_dtype
        self.keep_names = keep_names
        self.rename_dict = rename_dict
        self.to_path = to_path

    def extract_bigcsv_item_info(self, input_file, **kw):
        '''
        kw = dict(sep=',')
        kw.update(dict(usecols=sel_names, dtype=s_dtype,
                       error_bad_lines=False, warn_bad_lines=False))
        '''
        import os
        import pandas as pd
        import pickle
        dirname = os.path.join(self.to_path, 'INFO')
        # dirname = os.path.join(to_path, 'INFO')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        kw.update(dict(usecols=self.sel_names, dtype=self.s_dtype,
                       error_bad_lines=False, warn_bad_lines=False))
        chunks = chunk_bigfile(input_file, **kw)
        data = pd.concat(chunks, ignore_index=True)  # aa2 = data[:50]
        # default axis=0, join='outer',
        # data = data.dropna(axis=0, how='any', subset=['SERIAL_NUMBER'])
        data = data.dropna(axis=0, how='any', subset=self.keep_names)
        data = data.reset_index(drop=True)
        data.columns
        data = data.rename(columns=self.rename_dict)
        data.columns  # aa = data[:10]
        if data.shape[0] == 0:
            return
        slim_path = os.path.join(self.to_path, 'SlimData')
        # slim_path = os.path.join(to_path, 'SlimData')
        if not os.path.exists(slim_path):
            os.makedirs(slim_path)
        fname = os.path.split(input_file)[1]
        save_name = os.path.join(slim_path, fname[:-4] + '.pkl')
        with open(save_name, 'wb') as handle:  # protocol=4
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        st_list = [s for s in set(data['STATION_NAME']) if isinstance(s, str)]
        st_list = sorted(st_list)
        data.shape
        data.columns
        for station_name in st_list:  # station_name = st_list[-1]
            # station_name = 'AT'
            df = data[data['STATION_NAME'] == station_name]
            df = df.reset_index(drop=True)
            df = df.drop(['STATION_NAME'], axis=1)
            extract_item_info(df, dirname, input_file, station_name)
            # aa = df[:10]
        df_rm = data[~data['STATION_NAME'].isin(st_list)]
        if len(df_rm) > 0:
            filename2 = os.path.join(dirname, 'UNKNOWN_' + fname[:-4] + '.csv')
            df_rm.to_csv(filename2, index=False, encoding='utf_8_sig')
            # 解决中文乱码问题
            filename2 = os.path.join(dirname, 'UNKNOWN_' + fname[:-4] + '.pkl')
            with open(filename2, 'wb') as handle:  # protocol=4
                pickle.dump(df_rm, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_pkl_save_Xtable(self, input_csv, item_ID, **kw):
        import os
        import pandas as pd
        import pickle
        import glob
        dirname = os.path.join(self.to_path, 'SlimData')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        kw.update(dict(usecols=self.sel_names,
                       error_bad_lines=False, warn_bad_lines=False))
        fname = os.path.split(input_csv)[1]
        print(fname)  # 'EVA-AL10_DBC_201607.csv' # in the csvfile
        # tit_name = '_'.join(fname.split('_')[:-1])  # remove the last one
        out_path = os.path.join(dirname, fname[:-4])
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        # rf = open(input_csv, 'r')
        with open(input_csv, 'r') as rf:
            chunkSize = 1000000
            reader = pd.read_csv(rf, chunksize=chunkSize, **kw)
            # can only read one zip file.
            # #### reader.get_chunk(chunkSize)
            # not this reader = pd.read_csv(rf, **kw, iterator=True)
            i = 1
            for chunk in reader:  # iterator
                chunk = chunk.dropna(axis=0, how='any', subset=self.keep_names)
                Xsn, Xt = chunk_2_Xtable(chunk, item_ID)
                # 数据中有不同的日期  # len(item_ID)
                out_file = out_path + "/Xdata_{}.pkl".format(i)
                with open(out_file, "wb") as f:
                    pickle.dump(Xt, f, protocol=pickle.HIGHEST_PROTOCOL)
                filename = out_path + "/SNdata_{}.pkl".format(i)
                with open(filename, 'wb') as handle:
                    pickle.dump(Xsn, handle, protocol=4)
                i += 1
        # the next step, read the pickles and append each pickle to
        # the desired dataframe.
        # Same Path as out_path i.e. where the pickle files are.
        pickle_path = out_path
        f_list = glob.glob(pickle_path + "/Xdata_*.pkl")
        # pd.read_pickle(filename); pickle.load(handle)
        # add it to a list; then pd.concat(thelist)
        # df = df.append(chunk, ignore_index=True) is very slooowly.
        chunks = []
        for i in range(len(f_list)):
            with open(f_list[i], 'rb') as handle:
                chunk = pickle.load(handle)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        zname = 'X_' + fname[:-4] + '.pkl'
        filename = os.path.join(dirname, zname)
        with open(filename, 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # #### to save all the pkl data
        f_list = glob.glob(pickle_path + "/SNdata_*.pkl")
        chunks = []
        for i in range(len(f_list)):
            with open(f_list[i], 'rb') as handle:
                chunk = pickle.load(handle)
            chunks.append(chunk)
        ds = pd.concat(chunks, ignore_index=True)
        zname = 'X_' + fname[:-4] + '_SN_ITEM_SIZE.pkl'
        filename = os.path.join(dirname, zname)
        with open(filename, 'wb') as handle:
            pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)







