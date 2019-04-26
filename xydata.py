# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:42:28 2019

@author: HaiyanJiang
@email : jianghaiyan.cn@gmail.com

"""


def summary_XY01data(data):
    from .utils import diff
    # j = data.columns[0]
    v_names = [j for j in data.columns if j.startswith('R3_') and '_R2_' in j]
    h_names = diff(data.columns, v_names)
    sm = dict(
        n_shape=data.shape,
        n_vars=len(v_names), n_heads=len(h_names),
        n_sn=len(set(data['SERIAL_NUMBER']))
        )
    return sm


def construct_XYdata(data_path, doc_from, doc_tmp, ntype):
    # data_path = r'D:/HWATdata2018'; ntype = 'nor'
    # doc_from = 'XtableNumFold_Cleaned'
    # doc_tmp = 'XY01data'
    import os
    import glob
    import pickle
    from .utils import diff, write_iter2txt
    to_path = os.path.join(data_path, doc_tmp)
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    pre_path = os.path.join(data_path, 'PreData')
    filename = os.path.join(pre_path, 'data_items_' + ntype + '_station.pkl')
    if not os.path.exists(filename):
        return
    with open(filename, 'rb') as handle:
        data_items = pickle.load(handle)
    xt_path = os.path.join(data_path, doc_from)
    for k in data_items:
        print(k)  # k is the station name
        flist = glob.glob(xt_path + '/' + k + '*' + ntype + '*Xtable*.pkl')
        filename = flist[0]  # MUST ONLY ONE
        with open(filename, 'rb') as handle:
            df = pickle.load(handle)
        dg = df.groupby(['SERIAL_NUMBER']).size()
        dx = dg.reset_index().rename(columns={0: 'count'})
        # this can be totally seperate data into df_uni and df_dup
        ds = dx[dx['count'] > 1]  # special,
        df_uni = df[~df['SERIAL_NUMBER'].isin(ds['SERIAL_NUMBER'])]
        # df_dup = df[df['SERIAL_NUMBER'].isin(ds['SERIAL_NUMBER'])]
        data = df_uni
        data = data.reset_index(drop=True)
        # to get the summary of the data
        sm = summary_XY01data(data)
        # fname = os.path.split(filename)[1][:-4]
        fname = k + '_' + ntype + '_XYdata'
        txt_name = os.path.join(to_path, 'summary_' + fname + '.txt')
        write_iter2txt(sm, txt_name)
        h_names = ['SERIAL_NUMBER', 'Y']
        d_names = diff(data.columns, h_names)  # 'Y' in d_names
        cnames = h_names + d_names
        if ntype == 'exp':
            data['Y'] = 1
        elif ntype == 'nor':
            data['Y'] = 0
        data = data[cnames]  # aa = data[:10]
        sname = os.path.join(to_path, fname + '.pkl')
        with open(sname, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def combine_XYdata(data_path, doc_tmp, doc_to):
    # data_path = r'D:/HWATdata2018'
    # doc_to ='XYdata'
    import os
    import glob
    import pickle
    import pandas as pd
    to_path = os.path.join(data_path, doc_to)
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    pre_path = os.path.join(data_path, 'PreData')
    filename = os.path.join(pre_path, 'data_items_nor_station.pkl')
    with open(filename, 'rb') as handle:
        data_items = pickle.load(handle)
    xt_path = os.path.join(data_path, doc_tmp)
    for k in data_items:
        print(k)
        v_list = []
        chunks = []
        flist = glob.glob(xt_path + '/' + k + '*XYdata.pkl')
        # filename = flist[0]
        for filename in flist:  # MUST ONLY TWO
            print(filename)
            with open(filename, 'rb') as handle:
                chunk = pickle.load(handle)
            chunks.append(chunk)
            v_list.append(set(chunk.columns.tolist()))
        data = pd.concat(chunks, join='inner', ignore_index=True)
        # vv = list(set.intersection(*v_list))
        sname = os.path.join(to_path, k + '_XYdata.pkl')
        with open(sname, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        sname = os.path.join(to_path, k + '_XYdata.csv')
        data.to_csv(sname, index=False)


class ConstructXYdata():

    def __init__(self, data_path, doc_from, doc_tmp, doc_to):
        self.data_path = data_path
        self.doc_from = doc_from
        self.doc_tmp = doc_tmp
        self.doc_to = doc_to

    def xydata(self):
        # ntype='exp' or ntype='nor'
        data_path = self.data_path
        doc_from = self.doc_from
        doc_tmp = self.doc_tmp
        doc_to = self.doc_to
        construct_XYdata(data_path, doc_from, doc_tmp, ntype='nor')
        construct_XYdata(data_path, doc_from, doc_tmp, ntype='exp')
        combine_XYdata(data_path, doc_tmp, doc_to)


if __name__ == '__main__':
    # cxy = ConstructXYdata(data_path, doc_from, doc_tmp, doc_to)
    # cxy.xydata()
    """
    cxy = ConstructXYdata(data_path, doc_from, doc_tmp, doc_to)
    cxy.xydata()
    """
    print(666, 'DONE!!!')



