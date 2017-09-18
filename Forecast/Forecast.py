import pandas as pd
pd.options.mode.chained_assignment = None
import pymc3 as pm
import numpy as np
from matplotlib import pyplot as plt  
from IPython.core.pylabtools import figsize 
import matplotlib.ticker as mtick
from datetime import datetime 
import matplotlib.dates as mdates

#--------------将CSV文件读入DataFrame--------------------
df1 = pd.read_csv("data/train/15_data.csv")

#获取时间time的那一列,这个冒号的意思是所有行，逗号表示行与列的区分
time=df1.loc[:,'time'] 
power=df1.loc[:,'power'] 

#将时间str格式转化为datetime格式
time_new = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in time]

#获取温度，格式不用转化
temperature=df1.loc[:,'environment_tmp']

#------------------获取结冰开始、结束时间---------------------------------
df2 = pd.read_csv("data/train/15_failureInfo.csv")
Ice_startTime=df2.loc[:,'startTime'] 
Ice_endTime=df2.loc[:,'endTime']

#转换成datetime格式
Ice_startTime_new = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in Ice_startTime]
Ice_endTime_new=[datetime.strptime(d,"%Y-%m-%d %H:%M:%S") for d in Ice_endTime]

#------------------获取不结冰开始、结束时间-------------------------------
df3 = pd.read_csv("data/train/15_normalInfo.csv")
NotIce_startTime=df3.loc[:,'startTime'] 
NotIce_endTime=df3.loc[:,'endTime']

#转换成datetime格式
NotIce_startTime_new = [datetime.strptime(d, "%Y/%m/%d %H:%M:%S") for d in NotIce_startTime]
NotIce_endTime_new=[datetime.strptime(d,"%Y/%m/%d %H:%M:%S") for d in NotIce_endTime]

#-------------------对结冰、不结冰的时间范围数据进行处理，合并----------------------------------
#将数据存入DataFrame对象中
df4=pd.DataFrame(Ice_startTime_new)
df5=pd.DataFrame(Ice_endTime_new)
df6=pd.DataFrame(NotIce_startTime_new)
df7=pd.DataFrame(NotIce_endTime_new)

#将时间数据进行合并
df8 = pd.concat([df4, df5,df6,df7],ignore_index=True)

Ice_len=len(Ice_startTime_new)+len(Ice_endTime_new)#结冰的时间长度
NotIce_len=len(NotIce_startTime_new)+len(NotIce_endTime_new)#不结冰的时间长度

#定义一个list，值为1或0，代表结冰或者不结冰，默认为0
t_Ice=[0 for i in range(1) for row in range(Ice_len+NotIce_len)]

#定义一个函数，判断时间t时，是否结冰。对列表_tIce的结冰范围的值改为1，其余部分值为0 
def t_IceOrNot(Ice_len,NotIce_len,t_Ice):
    i=0
    while i<(Ice_len+NotIce_len):
        if i<Ice_len:
            t_Ice[i]=1
            i=i+1
        else:
            t_Ice[i]=0
            i=i+1
    return t_Ice
t_Ice_new=t_IceOrNot(Ice_len,NotIce_len,t_Ice)

#----------------------将结冰时间t与是否结冰I数据进行合并，横向------------------
df9 =pd.DataFrame(t_Ice_new)
df10 = pd.concat([df8, df9], axis=1)
df10.columns=['time','ice_or_not']#添加表头方便排序
#将数据进行排序
df11=df10.sort_values('time')

#写入文件newdata.csv    #不要索引和表头
df11.to_csv("data/train/15_newdata.csv",index=False)
df12 = pd.read_csv("data/train/15_newdata.csv")

#读取排序后的时间、以及是否结冰，进行画图
sort_time_data=df12.loc[:,"time"]
sort_time_data_new = [datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in sort_time_data]
sort_ice_data=df12.loc[:,"ice_or_not"]

#---------------------画图（1），线条（1）：时间t时是否结冰I，紫色（2）时间t时温度T的变化，红色----------------------
fig,ax1 = plt.subplots(1,1,figsize=(16.5,10))
# 设置X轴的坐标刻度线显示间隔
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d %H:%M:%S'))#设置时间标签显示格式
ax1.plot(sort_time_data_new,sort_ice_data,color="#7A68A6",alpha=0.85,label="t-Ice")
ax1.plot(time_new,temperature,color="#A60628",alpha=0.85,label="t-T")
ax1.legend(loc="upper right")
ax1.set_xlabel("Time,$t$")
ax1.set_ylabel("Temperature-Ice,$T-I$")
ax1.set_title("Time-Temperature,Ice")
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, hspace=0.2, wspace=0.3)#设置上下左右空白大小
plt.xticks(rotation=20)
plt.show()

#----------------------------

#def T_IceOrNot(time_new,Ice_startTime_new,Ice_endTime_new,T_Ice):
#    i,j=0,0
#    while(i<len(time_new)):
#        while(j<len(Ice_startTime_new)):
#            if(Ice_startTime_new[j]<=time_new[i]<=Ice_endTime_new[j]):
#                T_Ice[i]=1
#            else:
#                T_Ice[i]=0
#            j=j+1
#        if(j==len(Ice_startTime_new)):
#                j=0
#        i=i+1
#    return T_Ice
#T_Ice_new=T_IceOrNot(time_new,Ice_startTime_new,Ice_endTime_new,T_Ice)
#T_Ice_new[392934:393886]#正确
#T_Ice_new[55065:56021]#错误
#T_Ice_new[6588:7031]#错误
#T_Ice_new

#定义一个list，值为1或0，代表结冰或者不结冰，默认为0
T_Ice=[0 for i in range(1) for row in range(len(temperature))]

##定义一个函数，判断温度T时，是否结冰。对列表T_Ice的结冰范围的值改为1，其余部分值为0 
def T_IceOrNot(time_new,Ice_startTime_new,Ice_endTime_new,T_Ice):
    for i in range(len(time_new)-1,-1,-1):
        for j in range(len(Ice_startTime_new)-1,-1,-1):
            if(Ice_startTime_new[j]<time_new[i]<=Ice_endTime_new[j]):
                T_Ice[i]=1
                break
            #else:
            #    T_Ice[i]=0        
        #if(j==0):
        #    j=len(Ice_startTime_new)-1
    return T_Ice
T_Ice_new=T_IceOrNot(time_new,Ice_startTime_new,Ice_endTime_new,T_Ice)
T_Ice_new[392934:393886]#错误
T_Ice_new[55065:56021]#错误
T_Ice_new[6586:7031]#正确


#----------------------画图（2），温度T时是否结冰I，蓝色--------------------
#先画前10000个
fig, ax1 = plt.subplots(1,1,figsize=(16.5,10))
# 设置X轴的坐标刻度线显示间隔
#ax1.plot(temperature[48000:51000],T_Ice_new[48000:51000],color="#00C5CD",alpha=0.5,label="t-Ice")
#ax1.scatter(power,T_Ice_new,color="#00C5CD",alpha=0.5,label="t-Ice",s=75)

ax1.scatter(temperature,T_Ice_new,color="#00C5CD",alpha=0.5,label="t-Ice",s=75)
ax1.legend(loc="upper right")
ax1.set_xlabel("Temperature,$T$")
ax1.set_ylabel("Ice,$I$")
ax1.set_title("Temperature-Ice")
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, hspace=0.2, wspace=0.3)#设置上下左右空白大小
plt.show()
temperature

#-------------------建立模型，采样，画出后验分布图--------------
#np.dot(）去掉     #定义逻辑函数p(t)
def logistic(x,beta,alpha=0):
    return 1.0/(1.0+np.exp(beta*x+alpha))

with pm.Model() as Ice_model:
    #alpha、beta的先验分布
    alpha=pm.Normal('alpha',mu=0,tau=0.001)
    beta=pm.Normal('beta',mu=0,tau=0.001)

    p_=logistic(temperature[:50000],beta,alpha)
    #观测值加入模型
    observed=pm.Bernoulli('obs',p=p_,observed=T_Ice_new[:50000])

#采样
from scipy import optimize
with Ice_model:
    start=pm.find_MAP(fmin=optimize.fmin_powell)
    step=pm.Slice()
    trace = pm.sample(1200,step=step,start=start)

#绘制样本直方图和值
pm.traceplot(trace)
plt.show()

#----------------------对既定的温度取值的期望概率，即对所有后验样本取均值，得到p(ti)--------------
figsize(12.5,4)
alpha_trace = trace['alpha']
beta_trace = trace['beta']

#linspace函数可以生成元素为50的等间隔数列
t=np.linspace(temperature.min()-1,temperature.max()+1,50)[:,None]
p_t=logistic(t,beta_trace,alpha_trace)
mean_prob_t=p_t.mean(axis=1)

plt.plot(t,mean_prob_t,lw=3,label="average posterior \nprobability of the ice")
plt.plot(t,p_t[:,0],ls="--",label="realization from posterior")
plt.plot(t,p_t[:,-2],ls="--",label="realization from posterior")

#plt.scatter(temperature,New_T_Ice,color="k",s=50,alpha=0.5)
plt.title("Posterior expected value of the probability of defect,including two realizations")
plt.legend(loc="lower left")
plt.ylim(-0.1,1.1)
plt.xlim(t.min(),t.max())
plt.xlabel("Temperature")
plt.ylabel("Probability")
plt.show()

##------------------画期望值曲线和每个点对应的95%的置信区间（CI）--------------------
from scipy.stats.mstats import mquantiles

qs=mquantiles(p_t.T,[0.025,0.975],axis=0)#p_t.T转置
plt.fill_between(t[:,0],*qs,color="#7A68A6",alpha=0.7)#颜色填充，在？？？
plt.plot(t[:,0],qs[0],lw=2,color="#7A68A6",alpha=0.7,label="95%CI")

plt.plot(t,mean_prob_t,lw=2,color="#348ABD",label="average posterior \nprobability of the defect")

plt.scatter(temperature,T_Ice_new,color="#A60628",s=50,alpha=0.5)#画散点图

plt.title("Posterior probability of estimates,given temperature $t$")
plt.legend(loc="lower left")
plt.ylim(-0.02,1.02)
plt.xlim(t.min(),t.max())
plt.xlabel("Temperature,$t$")
plt.ylabel("Probability estimate")
plt.show()

#------------------定义通用的类------------------
class Temperature_Ice:
    a=[[]]
     #定义构造方法
    def __init__(self,temperature,New_T_Ice):
        self.a=[temperature,New_T_Ice]

Ice=Temperature_Ice(temperature,New_T_Ice)
print(Ice.a)

