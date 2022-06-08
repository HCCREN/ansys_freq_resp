#Import Module===============================================
import numpy as np
import control as co
import pandas as pd
import plotly.graph_objects as go
from scipy import signal
from scipy.fft import fft, fftfreq
import plotly.express as px
import re
#Import Ansys data===============================================
freq = pd.read_csv('FREQ_wo_elbox.txt',header=None)
freq = freq.rename(columns={0:'freq'})
np_freq = freq.to_numpy()*2*np.pi
rsdl = pd.read_csv('RSDL_wo_elbox.txt',sep="   ",header=None)
rsdl = rsdl.rename(columns={0:'X',1:'Z',2:'Y'})
np_rsdl = rsdl.to_numpy()/1.6

#Force response============================================
##Build transfer function:np_rsdl、np_freq=================
mon_A = 4  #m/s^2
def modeSum(np_rsdl,np_freq,xyz_021, tot_Ti,tot_Ax):
    t = tot_Ti/1000
    u = tot_Ax/1000 #mm/s^2 -> m/s^2 
    DR = 0.0329
    mode = []
    for i in range(len(np_rsdl)):
        G1 = co.tf([np_rsdl[i][xyz_021]], [1, 2*DR*np_freq[i][0], np_freq[i][0]*np_freq[i][0]])/mon_A
        t1, y1 = co.forced_response(G1, t, u)
        mode.append(y1)
    modeSum = sum(mode)
    return modeSum, t1

modeSum_Z = modeSum(np_rsdl,np_freq,1,tot_Ti,tot_Ax)
modeSum_X = modeSum(np_rsdl,np_freq,0,tot_Ti,tot_Ax)
modeSum_Y = modeSum(np_rsdl,np_freq,2,tot_Ti,tot_Ax)
#Read KGM===================================================
file_name = 'z2y_F10000_P1660_1600_P1772_8_P1769_0.txt'
df_kgm = pd.read_csv(file_name,sep='delimiter', header=None, engine='python')
sep_column = 2
df_kgm = df_kgm[0].str.split('\t', sep_column, expand=True)
result = np.where(np.array(df_kgm[2]) == df_kgm[0][55])
for i in range(result[0][0]-1):   #將含%[DATA]以前的行數全部刪除
  df_kgm  = df_kgm .drop([i])
line = 'Z'
cross = 'Y'
df_kgm = df_kgm.astype(float)
df_kgm.columns =[cross,line,'Freq']
df_kgm = df_kgm[:15000]
df_kgm = df_kgm.reset_index()
df_kgm = df_kgm.drop(['index'],axis=1)
df_kgm['Freq'] = df_kgm['Freq']/1000000
#Calculate velocity
#df_kgm['Z']=df_kgm['Z']-df_kgm['Z']
count_row1 = df_kgm[line].shape[0]    #找出總共有幾行
length1 = count_row1
df_kgm['V']=np.zeros(length1)
df_kgm['A']=np.zeros(length1)
df_kgm['J']=np.zeros(length1)
df_kgm['Time']=np.zeros(length1)
for i in range(length1-1):
  df_kgm['V'][i+1] = ((df_kgm[line][i+1] - df_kgm[line][i]) / df_kgm['Freq'][i]) * 60 # 將mm/s變成 mm/min
  df_kgm['A'][i+1] = ((df_kgm['V'][i+1] - df_kgm['V'][i]) / df_kgm['Freq'][i]) /60
  df_kgm['J'][i+1] = ((df_kgm['A'][i+1] - df_kgm['A'][i]) / df_kgm['Freq'][i])
  df_kgm['Time'][i+1] = df_kgm['Time'][i]+df_kgm['Freq'][i+1]
Z_refine = np.zeros(len(df_kgm['Y']))
kgm_stop_idx = np.searchsorted(df_kgm['Time'], Stop_Ti/1000, side="left")
for i in range(kgm_stop_idx):
  Z_refine[i] = i*0.01/kgm_stop_idx
Z_refine[kgm_stop_idx:15000] = 0.01

#Plot=======================================================
fig = go.Figure()
#fig =px.line(x=t1,y=np.sum(mode,axis=0),width=600, height=600)
sim_Y = modeSum_Y[0]*1000
fig.add_trace(go.Scatter(x=modeSum_Y[1], y=sim_Y, name='simulation',
                         line=dict(color='firebrick', width=4)))
fig.add_trace(go.Scatter(x=df_kgm['Time'], y=df_kgm['Y']-Z_refine, name = 'real',
                         line=dict(color='royalblue', width=4)))
fig.update_xaxes(title_text='Time(s)')
fig.update_yaxes(title_text='displacement(m)')
fig.update_layout(
    title='sim_vibration vs real_vibration without ELBOX '+para,
    yaxis = dict(
        showexponent = 'all',
        exponentformat = 'e',
        tickfont_family="Arial"
    )
    ,font=dict(
    size=18,
    color="black"
    )
)

fig.show()

#FFT=======================================================
def fft_kgm(target_wave,target_time,title_name):
  
  y1 = list(target_wave)
  x1 = list(target_time)
  N = len(y1)
  T = x1[20]-x1[19]
  yf = fft(y1)
  xf = fftfreq(N,T)[:N//2]
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=xf, y=2.0/N * np.abs(yf[0:N//2]),
                    mode='lines',
                    name='lines'))
  # ax = plt.gca()
  # miloc = plt.MultipleLocator(10)
  # #ymiloc = plt.MultipleLocator(10)
  # ax.xaxis.set_minor_locator(miloc)
  # #ax.yaxis.set_minor_locator(ymiloc)
  # plt.grid(which = 'minor')
  # Edit the layout
  fig.update_layout(title=title_name,
                    xaxis_title='Freq',
                    yaxis_title='Amp',
                    yaxis = dict(
                        showexponent = 'all',
                        exponentformat = 'e',
                        tickfont_family="Arial",
                    )
                    ,font=dict(
                        size=18,
                        color="black"
                    )
                    )
                   
  fig.update_layout(xaxis_range=[0,500])
  fig.update_layout(yaxis_range=[0,0.001])
  
  fig.show() 
  fft_y = 2.0/N * np.abs(yf[0:N//2])  

  dB_fft = 20*np.log10(fft_y)
  fft1time = xf
  return fft_y,dB_fft,fft1time,T

target_wave = sim_Y
target_time = modeSum_Y[1]
fft1 = fft_kgm(target_wave,target_time,'simulate_paht')

target_wave = df_kgm['Y']-Z_refine
target_time = df_kgm['Time']
fft1 = fft_kgm(target_wave,target_time,'real_path')
