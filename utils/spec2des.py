import pandas as pd
import numpy as np
from rdkit import Chem


def drop_zero(df):
    return df.loc[:, (df != 0).any(axis=0)].loc[:, (df != None).any(axis=0)]


def drop_high_correlation_col(data, threshold=0.9):
    # Create correlation matrix
    corr_matrix = data.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than threshold(default=0.9)
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Drop features
    new_data = data.drop(to_drop, axis=1)
    new_data.dropna(axis=1)
    return new_data


def add_ir_feature(features, row, peak=None, width=None, begin=None, end=None):
    if not peak:
        peak = (end + begin) / 2
    if not width:
        width = abs(end - begin) / 2
    try:
        features['IR_{}~{}'.format(peak, width)]
    except:
        features['IR_{}~{}'.format(peak, width)] = 0
    if peak-width < row['x'] < peak+width and row['y'] > 0.05 and row['y'] > features['IR_{}~{}'.format(peak, width)]:
        features['IR_{}~{}'.format(peak, width)] = row['y']
        return True
    return False

'''
def format_ir_no_exp(cas_map, digit_spec_path, select_col='cas'):
    formated_IR = []
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        ir = pd.read_csv(
            '{}/{}_IR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        print(cas)
        ir['y'] = 1 - (ir['y']-ir['y'].min())/(ir['y'].max()-ir['y'].min())
        features = {}
        for index_ir, row_ir in ir.iterrows():
            for i in range(3500, 500, -20):
                if add_ir_feature(features, row_ir, i-10, 10):
                    break
        formated_IR.append(list(features.values()))
    formated_IR = pd.DataFrame(formated_IR)
    formated_IR.columns = features.keys()
    formated_IR[select_col] = cas_map[select_col]
    return formated_IR
'''

def format_ir_no_exp(cas_map, digit_spec_path, select_col='cas', name='_', start=500.0, end=3500.0, interval=20.0):
    ls = [i for i in np.arange(start, end+interval, interval)]
    label = ['{}_IR_{:.4}~{:.4}'.format(name, i, i+interval) for i in np.arange(start, end, interval)]
    # print(len(ls), len(label))
    formated_IR = pd.DataFrame(columns=label)
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        ir = pd.read_csv(
            '{}/{}_IR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        # print(cas)
        ir['feature'] = pd.cut(ir['x'], ls, labels=label)
        result = ir[['feature', 'y']].groupby('feature').agg('min')
        result = 1 - result/result.max()
        result = result.rename(columns={'y':0}).T
        result[select_col] = cas
        formated_IR = formated_IR.append(result)
    return formated_IR.dropna(axis=1, how='all').fillna(0)


def format_ir_exp(cas_map, digit_spec_path, select_col='cas'):
    formated_IR = []
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        ir = pd.read_csv(
            '{}/{}_IR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        print(cas)
        ir['y'] = 1 - (ir['y']-ir['y'].min())/(ir['y'].max()-ir['y'].min())
        features = {}
        for index_ir, row_ir in ir.iterrows():
            add_ir_feature(features, row_ir, 2962, 10)
            add_ir_feature(features, row_ir, 2872, 10)
            add_ir_feature(features, row_ir, 2926, 10)
            add_ir_feature(features, row_ir, 2853, 10)
            add_ir_feature(features, row_ir, 2890, 10)
            add_ir_feature(features, row_ir, 3025, 15)
            add_ir_feature(features, row_ir, 3300, 10)
            add_ir_feature(features, row_ir, 3050, 50)
            add_ir_feature(features, row_ir, 2185, 85)
            add_ir_feature(features, row_ir, None, None, 1540, 1695)
            add_ir_feature(features, row_ir, None, None, 1667, 2000)
            add_ir_feature(features, row_ir, None, None, 1430, 1650)
            add_ir_feature(features, row_ir, 1715, 10)
            add_ir_feature(features, row_ir, 1450, 10)
            add_ir_feature(features, row_ir, 1375, 5)
            add_ir_feature(features, row_ir, 1465, 20)
            add_ir_feature(features, row_ir, 1340, 10)
            add_ir_feature(features, row_ir, None, None, 1295, 1310)
            add_ir_feature(features, row_ir, None, None, 665, 770)
            add_ir_feature(features, row_ir, None, None, 750, 600)
            add_ir_feature(features, row_ir, None, None, 960, 970)
            add_ir_feature(features, row_ir, None, None, 770, 730)
            add_ir_feature(features, row_ir, None, None, 770, 735)
            add_ir_feature(features, row_ir, None, None, 810, 750)
            add_ir_feature(features, row_ir, None, None, 860, 840)
            add_ir_feature(features, row_ir, None, None, 725, 680)
            add_ir_feature(features, row_ir, None, None, 825, 775)
            add_ir_feature(features, row_ir, None, None, 900, 860)
            add_ir_feature(features, row_ir, None, None, 860, 780)
            add_ir_feature(features, row_ir, None, None, 920, 800)
            add_ir_feature(features, row_ir, None, None, 910, 665)
            add_ir_feature(features, row_ir, None, None, 1000, 960)
            add_ir_feature(features, row_ir, None, None, 1390, 1315)
            add_ir_feature(features, row_ir, 2820, 10)
            add_ir_feature(features, row_ir, 2720, 10)
            add_ir_feature(features, row_ir, 3450, 10)
            add_ir_feature(features, row_ir, 3350, 10)
            add_ir_feature(features, row_ir, 3180, 10)
            add_ir_feature(features, row_ir, 3270, 10)
            add_ir_feature(features, row_ir, 3030, 10)
            add_ir_feature(features, row_ir, None, None, 3060, 3010)
            add_ir_feature(features, row_ir, None, None, 1740, 1690)
            add_ir_feature(features, row_ir, None, None, 2260, 2240)
            add_ir_feature(features, row_ir, 1740, 10)
            add_ir_feature(features, row_ir, None, None, 1450, 1410)
            add_ir_feature(features, row_ir, None, None, 1266, 1205)
            add_ir_feature(features, row_ir, None, None, 1850, 1880)
            add_ir_feature(features, row_ir, None, None, 1780, 1740)
            add_ir_feature(features, row_ir, None, None, 1770, 1720)
            add_ir_feature(features, row_ir, None, None, 1670, 1630)
            add_ir_feature(features, row_ir, None, None, 1680, 1630)
            add_ir_feature(features, row_ir, None, None, 1650, 1590)
            add_ir_feature(features, row_ir, None, None, 1650, 1550)
            add_ir_feature(features, row_ir, None, None, 1680, 1650)
            add_ir_feature(features, row_ir, None, None, 1810, 1790)
            add_ir_feature(features, row_ir, None, None, 1565, 1543)
            add_ir_feature(features, row_ir, None, None, 1550, 1510)
            add_ir_feature(features, row_ir, None, None, 1580, 1520)
            add_ir_feature(features, row_ir, None, None, 1170, 1050)
            add_ir_feature(features, row_ir, None, None, 1340, 1250)
            add_ir_feature(features, row_ir, None, None, 1350, 1280)
            add_ir_feature(features, row_ir, None, None, 1300, 1000)
            add_ir_feature(features, row_ir, None, None, 1420, 1400)
            add_ir_feature(features, row_ir, None, None, 1385, 1360)
            add_ir_feature(features, row_ir, None, None, 1365, 1335)
            add_ir_feature(features, row_ir, None, None, 1175, 1000)
            add_ir_feature(features, row_ir, None, None, 3500, 3300)
            add_ir_feature(features, row_ir, None, None, 1335, 1165)
            add_ir_feature(features, row_ir, None, None, 1230, 1010)
            add_ir_feature(features, row_ir, None, None, 1410, 1260)
            add_ir_feature(features, row_ir, None, None, 1250, 1000)
            add_ir_feature(features, row_ir, None, None, 1667, 1430)
            add_ir_feature(features, row_ir, None, None, 1650, 1250)
            add_ir_feature(features, row_ir, 3450, 250)
            add_ir_feature(features, row_ir, 3415, 290)
            add_ir_feature(features, row_ir, None, None, 3400, 2500)
        formated_IR.append(list(features.values()))
    formated_IR = pd.DataFrame(formated_IR)
    formated_IR.columns = features.keys()
    formated_IR[select_col] = cas_map[select_col]
    return formated_IR


def clean_xy(nmr_type, smi, spec_path='nova_spec/xy'):
    test_df = pd.read_csv('{}/{}/{}.csv'.format(spec_path, nmr_type, smi))
    test_df.columns = ['x', 'y']
    test_df['y'] = (test_df['y']-test_df['y'].min())/(test_df['y'].max()-test_df['y'].min())
    test_df = test_df.loc[test_df['y']>0.005].reset_index().drop('index', axis=1)
    return test_df


def add_hnmr_feature(features, row, peak=None, width=None, begin=None, end=None, name='', lower_limit=0.05):
    if not peak:
        peak = (end + begin) / 2
    if not width:
        width = abs(end - begin) / 2
    try:
        features['{}_HNMR_{:.3}~{:.3}'.format(name, peak, width)]
    except:
        features['{}_HNMR_{:.3}~{:.3}'.format(name, peak, width)] = 0
    if peak-width < row['x'] < peak+width and row['y'] > lower_limit:
        features['{}_HNMR_{:.3}~{:.3}'.format(name, peak, width)] += row['y']
        return True
    return False

'''
def format_hnmr_no_exp(cas_map, digit_spec_path, select_col='cas', name='_'):
    formated_HNMR = []
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        hnmr = pd.read_csv(
            '{}/{}_HNMR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        print(cas)
        hnmr_features = {}
        hnmr['y'] = (hnmr['y']-hnmr['y'].min()) / \
            (hnmr['y'].max()-hnmr['y'].min())
        for index_hnmr, row_hnmr in hnmr.iterrows():
            for i in np.arange(13.1, 0.6, -0.2):
                if add_hnmr_feature(hnmr_features, row_hnmr, i-0.1, 0.1, name=name):
                    break
        new_features = np.array(list(hnmr_features.values()))
        formated_HNMR.append(new_features / new_features.max())
    formated_HNMR = pd.DataFrame(formated_HNMR)
    formated_HNMR.columns = hnmr_features.keys()
    formated_HNMR[select_col] = cas_map[select_col]
    return formated_HNMR
'''

def format_hnmr_no_exp(cas_map, digit_spec_path, select_col='cas', name='', start=0.6, end=13.1, interval=0.2, lower_limit=0.05):
    ls = [i for i in np.arange(start, end+interval, interval)]
    label = ['{}_HNMR_{:.4}~{:.4}'.format(name, i, i+interval) for i in np.arange(start, end, interval)]
    # print(len(ls), len(label))
    formated_HNMR = pd.DataFrame(columns=label)
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        hnmr = pd.read_csv(
            '{}/{}_HNMR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        hnmr = hnmr.loc[hnmr['y']>lower_limit]
        # print(cas)
        hnmr['feature'] = pd.cut(hnmr['x'], ls, labels=label)
        result = hnmr[['feature', 'y']].groupby('feature').agg('sum')
        result = result/result.max()
        result = result.rename(columns={'y':0}).T
        result[select_col] = cas
        formated_HNMR = formated_HNMR.append(result)
    return formated_HNMR.dropna(axis=1, how='all').fillna(0)


def format_hnmr_exp(cas_map, digit_spec_path, select_col='cas', name='_'):
    formated_HNMR = []
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        hnmr = pd.read_csv(
            '{}/{}_HNMR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        print(cas)
        hnmr_features = {}
        hnmr['y'] = (hnmr['y']-hnmr['y'].min()) / \
            (hnmr['y'].max()-hnmr['y'].min())
        for index_hnmr, row_hnmr in hnmr.iterrows():
            flag = add_hnmr_feature(hnmr_features, row_hnmr, 0.9, 0.05, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(hnmr_features, row_hnmr, 1.3, 0.05, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(hnmr_features, row_hnmr, 1.5, 0.05, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(hnmr_features, row_hnmr, 1.7, 0.05, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2.4, 3.2, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2.2, 3, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2, 2.2, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2, 2.6, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2.1, 3.2, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2, 3.1, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2.7, 3.8, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 3.5, 4, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 4, 4.5, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 3.1, 3.5, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 3.7, 4, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 3.6, 4, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 3.4, 3.8, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 3.4, 4, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(hnmr_features, row_hnmr, None, None, 3, 5, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 4.5, 5.9, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 0.9, 2.5, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 0.5, 3.5, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 4.7, 7.7, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 0.5, 5.5, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 6.4, 9.5, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(hnmr_features, row_hnmr, None, None, 5, 9, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(hnmr_features, row_hnmr, None, None, 9, 10, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 11, 12, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 10, 13, name=name)
            if flag:
                continue
            flag = add_hnmr_feature(
                hnmr_features, row_hnmr, None, None, 2.9, 6.5, name=name)
            if flag:
                continue
        new_features = np.array(list(hnmr_features.values()))
        formated_HNMR.append(new_features / new_features.max())
    formated_HNMR = pd.DataFrame(formated_HNMR)
    formated_HNMR.columns = hnmr_features.keys()
    formated_HNMR[select_col] = cas_map[select_col]
    return formated_HNMR


def add_cnmr_feature(features, row, peak=None, width=None, begin=None, end=None, name='_'):
    if not peak:
        peak = (end + begin) / 2
    if not width:
        width = abs(end - begin) / 2
    try:
        features['{}_CNMR_{}~{}'.format(name, peak, width)]
    except:
        features['{}_CNMR_{}~{}'.format(name, peak, width)] = 0
    if peak-width < row['x'] < peak+width and row['y'] > 0.05:
        features['{}_CNMR_{}~{}'.format(name, peak, width)] += row['y']
        return True
    return False

'''
def format_cnmr_no_exp(cas_map, digit_spec_path, select_col='cas', name='_'):
    formated_CNMR = []
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        cnmr = pd.read_csv(
            '{}/{}_CNMR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        print(cas)
        cnmr_features = {}
        cnmr['y'] = (cnmr['y']-cnmr['y'].min()) / \
            (cnmr['y'].max()-cnmr['y'].min())
        for index_cnmr, row_cnmr in cnmr.iterrows():
            for i in np.arange(215, 5, -10):
                if add_cnmr_feature(cnmr_features, row_cnmr, i-5, 5, name=name):
                    break
        new_features = np.array(list(cnmr_features.values()))
        formated_CNMR.append(new_features / new_features.max())
    formated_CNMR = pd.DataFrame(formated_CNMR)
    formated_CNMR.columns = cnmr_features.keys()
    formated_CNMR[select_col] = cas_map[select_col]
    return formated_CNMR
'''

def format_cnmr_no_exp(cas_map, digit_spec_path, select_col='cas', name='', start=5.0, end=215.0, interval=10.0, lower_limit=0.05):
    ls = [i for i in np.arange(start, end+interval, interval)]
    label = ['{}_CNMR_{:.4}~{:.4}'.format(name, i, i+interval) for i in np.arange(start, end, interval)]
    formated_CNMR = pd.DataFrame(columns=label)
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        cnmr = pd.read_csv(
            '{}/{}_CNMR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        # print(cas)
        cnmr = cnmr.loc[cnmr['y']>lower_limit]
        cnmr['feature'] = pd.cut(cnmr['x'], ls, labels=label)
        result = cnmr[['feature', 'y']].groupby('feature').agg('max')
        result = result/result.max()
        result = result.rename(columns={'y':0}).T
        result[select_col] = cas
        formated_CNMR = formated_CNMR.append(result)
    return formated_CNMR.dropna(axis=1, how='all').fillna(0)


def format_cnmr_exp(cas_map, digit_spec_path, select_col='cas', name='_'):
    formated_CNMR = []
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        cnmr = pd.read_csv(
            '{}/{}_CNMR.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        print(cas)
        cnmr_features = {}
        cnmr['y'] = (cnmr['y']-cnmr['y'].min()) / \
            (cnmr['y'].max()-cnmr['y'].min())
        for index_cnmr, row_cnmr in cnmr.iterrows():
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 20, 45, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 40, 60, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 40, 70, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(cnmr_features, row_cnmr, 71.9, 1.0, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 60, 75, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 65, 90, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(cnmr_features, row_cnmr, None, None, 0, 70, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(cnmr_features, row_cnmr, 123.3, 1.0, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(cnmr_features, row_cnmr, 128.5, 1.0, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 120, 160, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 100, 150, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 160, 185, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 175, 205, name=name)
            if flag:
                continue
            flag = add_cnmr_feature(
                cnmr_features, row_cnmr, None, None, 200, 220, name=name)
            if flag:
                continue
        new_features = np.array(list(cnmr_features.values()))
        formated_CNMR.append(new_features / new_features.max())
    formated_CNMR = pd.DataFrame(formated_CNMR)
    formated_CNMR.columns = cnmr_features.keys()
    formated_CNMR[select_col] = cas_map[select_col]
    return formated_CNMR


def format_ms(cas_map, digit_spec_path, select_col='cas'):
    formated_MS = []
    for index, row in cas_map.iterrows():
        cas = row[select_col]
        ms = pd.read_csv(
            '{}/{}_MS.csv'.format(digit_spec_path, cas)).dropna(axis=1)
        ms_features = {}
        ms['y'] = (ms['y']-ms['y'].min())/(ms['y'].max()-ms['y'].min())
        print(cas)
        ms = ms.sort_values(by='y', ascending=False)
        i = 0
        for index_ms, row_ms in ms[:15].iterrows():
            i += 1
            ms_features['MS_top_{}'.format(i)] = np.around(
                row_ms['x'])*row_ms['y']
        new_features = np.array(list(ms_features.values()))
        formated_MS.append(new_features / new_features.max())
    formated_MS = pd.DataFrame(formated_MS)
    formated_MS.columns = ms_features.keys()
    formated_MS[select_col] = cas_map[select_col]
    return formated_MS


def format_smi(df, select_col):
    standard_smiles = []
    for index, row in df.iterrows():
        mol = Chem.MolFromSmiles(row[select_col])
        smi = Chem.MolToSmiles(mol)
        standard_smiles.append(smi)
    df[select_col] = standard_smiles
    return df


def dummies(df, select_col):
    col_dummies = pd.get_dummies(df[select_col], prefix=select_col)
    temp_data = df.drop(select_col, axis=1)
    temp_data = pd.concat([temp_data, col_dummies], axis=1)
    return temp_data


def onehot_des(df, select_col):   
    smiles = df[select_col].drop_duplicates(
        keep='first').to_frame().reset_index().drop('index', axis=1)
    temp_df = dummies(smiles, select_col)
    temp_df[select_col] = smiles[select_col]
    return temp_df