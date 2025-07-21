import pandas as pd
#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Text 파일내 Data를 사용 가능할수있게 변환하는 함수 모음

def arr_load(data, freq ,ha):
    d = ["","","",""]
    d[0] = float(data[4:17])
    d[1] = float(data[17:31])
    d[2] = float(data[31:45])
    d[3] = float(data[45:58])
    d.insert(0,freq)
    d.insert(0,ha)
    return d

def st_load(data,freq,ha):    
    d = ["","","","",""]    
    d[0] = float(data[1:3])
    if data[4:17] == '    -nan(ind)':
        d[1] = 0
    else:
        d[1] = float(data[4:17])    
    if data[17:31] == '    -nan(ind) ':
        d[2] = 0
    else:
        d[2] = float(data[17:31])    
    if data[31:45] == '    -nan(ind) ':
        d[3] = 0
    else:
        d[3] = float(data[31:45])    
    if data[45:58] == '    -nan(ind)':
        d[4] = 0
    else:
        d[4] = float(data[45:58])        
    d.insert(0,freq)
    d.insert(0,ha)
    return d

# def riTrans(data,datar_a,datai_a,ha):
#     datar = data['Real']  
#     datai = data['Imag'] 
#     datar = datar.reset_index()
#     datai = datai.reset_index()
#     datar = datar.drop(['index'],1)
#     datai = datai.drop(['index'],1)
#     datar.columns = ['{} deg'.format(i)]
#     datai.columns = ['{} deg'.format(i)]
#     datar_all = pd.concat([datar_a, datar],axis=1)
#     datai_all = pd.concat([datai_a, datai],axis=1)
#     return datar_all,datai_all

def riTrans(data, datar_a, datai_a, ha):
    # Series 형태인 경우 DataFrame으로 변환
    datar = data['Real'].to_frame() if isinstance(data['Real'], pd.Series) else data['Real']
    datai = data['Imag'].to_frame() if isinstance(data['Imag'], pd.Series) else data['Imag']
    
    # 인덱스 재설정
    datar = datar.reset_index(drop=True)
    datai = datai.reset_index(drop=True)
    
    # 열 이름 설정
    column_name = f'{ha} deg'
    datar.columns = [column_name]
    datai.columns = [column_name]
    
    # 데이터 합치기
    datar_all = pd.concat([datar_a, datar], axis=1)
    datai_all = pd.concat([datai_a, datai], axis=1)
    
    return datar_all, datai_all



def loadTrans(data):
    if type(data) == 'pandas.core.frame.DataFrame':
        data.reset_index(inplace=True)
        data = data.drop(['index'],1)
    else:
        data = pd.DataFrame(data)
        data.reset_index(inplace=True)
        data = data.drop(['index'],1)        
    return data


# In[ ]:




