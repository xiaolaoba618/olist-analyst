# -*- coding: utf-8 -*-
"""汇总各文件夹分析结果，供 PPT 报告取数"""
import pandas as pd

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)
BASE = r'G:/Documents/学习项目/数据分析/kaggle巴西电商分析/'
rd = lambda p, **kw: pd.read_csv(BASE + p, **kw)

print('###### 1. 延迟天数 × 评分（边际效应）')
d = rd('延迟率分析/processed_data/延迟天数与订单评分关系.csv')
print(d.head(12).to_string())
print('延迟订单总数:', d.loc[d['超时天数'] > 0, '该天数总订单数'].sum())

print('\n###### 2. 发货州维度')
s = rd('延迟率分析/processed_data/延迟率与发货州关系.csv', encoding='gbk')
print(s.sort_values('该州超时率', ascending=False).head(10).to_string())

print('\n###### 3. 商家分层（分级后总数据）')
t = rd('延迟率分析/processed_data/分级后总数据.csv')
print('总订单:', len(t), '| 超时订单:', t['最终是否超时'].sum(), '| 超时率: %.2f%%' % (t['最终是否超时'].mean() * 100))
print('平均评分: %.3f' % t['订单评分'].mean())
g = t.groupby('商家风险等级').agg(订单数=('订单评分', 'size'), 超时率=('最终是否超时', 'mean'), 评分=('订单评分', 'mean'))
g['超时率'] = (g['超时率'] * 100).round(2)
g['评分'] = g['评分'].round(2)
print(g.sort_values('订单数', ascending=False).to_string())
print('\n超时归因分布:')
print(t[t['最终是否超时'] == 1]['商家超时归因'].value_counts().to_string())

print('\n###### 4. 商品分层')
g2 = t.groupby('商品风险等级').agg(订单数=('订单评分', 'size'), 超时率=('最终是否超时', 'mean'), 评分=('订单评分', 'mean'))
g2['超时率'] = (g2['超时率'] * 100).round(2)
g2['评分'] = g2['评分'].round(2)
print(g2.sort_values('订单数', ascending=False).to_string())

print('\n###### 5. 商家层面结果文件（各类商家数）')
for f in ['A_卖家主体严重风险_建议下架评估.csv', 'B_送达超时非卖家责任_优化预测_快递.csv',
          'C_发货行为违规预警_约谈警告.csv', 'D_优质卖家_无需干预.csv']:
    try:
        x = rd('延迟率分析/商家层面分析结果/' + f)
        print(f, '→', len(x), '家')
    except Exception as e:
        print(f, 'ERR', e)

print('\n###### 6. 商品层面结果文件')
for f in ['A_优质商品_不易超时且评分高.csv', 'B_评分受超时率影响最大.csv',
          'C_警惕商品_高延迟但暂未严重影响评分.csv', 'D_普通商品_表现正常.csv']:
    try:
        x = rd('延迟率分析/商品层面分析结果/' + f)
        print(f, '→', len(x), '类')
    except Exception as e:
        print(f, 'ERR', e)

print('\n###### 7. 2018-02/03 异常期')
z = rd('延迟率分析/processed_data/2019异常分析.csv')
print('订单数:', len(z), '| 平均评分: %.2f' % z['订单评分'].mean(), '| 平均最终延迟: %.2f 天' % z['实际最终延迟时间'].mean())
print(z['商家超时归因'].value_counts(normalize=True).round(3).to_string())
print(z['商家超时归因'].value_counts().to_string())

print('\n###### 8. 全量对照（分级后总数据）归因占比')
print(t[t['最终是否超时'] == 1]['商家超时归因'].value_counts(normalize=True).round(3).to_string())

print('\n###### 9. RFM 客户表')
c = rd('客户分层RFM分析/output/c_output.csv')
print('客户数:', len(c), '| 复购人数:', (c['订单总数'] > 1).sum(),
      '| 复购率: %.2f%%' % ((c['订单总数'] > 1).mean() * 100))
print('平均消费: %.2f' % c['总消费额'].mean(), '| 中位数: %.2f' % c['总消费额'].median())

print('\n###### 10. 时间序列')
try:
    ts = rd('订单信息时间序列分析/output/output.csv', encoding='utf-8')
    print(ts.head(3).to_string())
    print(ts.tail(3).to_string(), '| shape', ts.shape)
except Exception as e:
    print('ERR', e)
