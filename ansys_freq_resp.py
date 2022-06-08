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

#Virtual CNC parameter setting======================================
P1660=1600
P1772=72
P1783=10000

P1735=0
P1732=10
P1769=0

P1737=0
P1738=10
para = 'z2y_F10000_P1660_'+str(P1660)+'_P1772_'+str(P1772)+'_P1769_0'
file_name = para+'.txt'
#Virtual CNC Controller===============================================
#function=============================================================
def cal(Ui, radius, feed, o_feed, parms, Link, Ti, Ax, Vx, Ay, Vy, Last_U, Last_D):
  #Input Data; parms = [F, P1660, P1772, P1783]
  #--------------------------------------------------------------------------------------
  pFl = feed[0]
  pF = feed[1]
  pFp = feed[2]
  pA = parms[0] # P1660
  pT = parms[1] # P1772
  pD = parms[2] # P1783

  ac_sup = parms[3] # P1735
  vc_inf = parms[4] # P1732
  delta_t = 0.001 # Time of P1783
  T_en = Ti[-1] # Beginning Time of each segment

  #Vector and Angle
  #--------------------------------------------------------------------------------------
  vec = np.array( [ Ui[1][0] - Ui[0][0] , Ui[1][1] - Ui[0][1] ], dtype=float)
  vecp = np.array( [ Ui[2][0] - Ui[1][0] , Ui[2][1] - Ui[1][1] ], dtype=float)
  theta = np.arctan2( vec[1], vec[0] )
  thetap = np.arctan2( vecp[1], vecp[0] )
  
  if vec[0] > 0.:
    Xsign = 1.
  elif vec[0] < 0.:
    Xsign = -1.
  else:
    Xsign = 0.
  
  if vec[1] > 0.:
    Ysign = 1.
  elif vec[1] < 0.:
    Ysign = -1.
  else:
    Ysign = 0.

  if vecp[0] > 0.:
    Xpsign = 1.
  elif vecp[0] < 0.:
    Xpsign = -1.
  else:
    Xpsign = 0.
  
  if vecp[1] > 0.:
    Ypsign = 1.
  elif vecp[1] < 0.:
    Ypsign = -1.
  else:
    Ypsign = 0.

  #Criteria
  #--------------------------------------------------------------------------------------
  #P1772
  if pT > 10**(-8):
    bool_P1772 = True
  else:
    pT = 0.0
    bool_P1772 = False

  #P1783
  maxV = max( abs( o_feed[1]*np.cos(theta)-o_feed[2]*np.cos(thetap) ), abs( o_feed[1]*np.sin(theta)-o_feed[2]*np.sin(thetap) ) )
  if maxV > pD and pD != 0. :
    Rc = maxV/pD
  else:
    Rc = 1.
  pD = o_feed[1]/Rc

  #pT
  if pT > delta_t:
    bool_pT = True
  else:
    bool_pT = False

  #P1735 & P1732  
  if radius[0] != 0 and np.sqrt(abs(radius[0])*ac_sup) < vc_inf and vc_inf < abs(pF):
    F_f = vc_inf
    bool_Ff = -1
  elif radius[0] != 0 and vc_inf < np.sqrt(abs(radius[0])*ac_sup) and np.sqrt(abs(radius[0])*ac_sup) < abs(pF):
    F_f = np.sqrt(abs(radius[0])*ac_sup)
    bool_Ff = -1
  elif radius[2] != 0 and np.sqrt(abs(radius[2])*ac_sup) < vc_inf and vc_inf < abs(pFp):
    F_f = vc_inf
    bool_Ff = 1
  elif radius[2] != 0 and vc_inf < np.sqrt(abs(radius[2])*ac_sup) and np.sqrt(abs(radius[2])*ac_sup) < abs(pFp):
    F_f = np.sqrt(abs(radius[2])*ac_sup)
    bool_Ff = 1
  else:
    F_f = abs(pF)
    bool_Ff = 0

  if radius[2] != 0 and pFp < pF:
    F_f = abs(pFp)
    bool_Ff = 1
  if radius[0] != 0 and pFl < pF:
    F_f = abs(pFl)
    bool_Ff = -1
  
  Fxf = F_f*np.cos(theta)
  Fyf = F_f*np.sin(theta)

  #Division 
  #--------------------------------------------------------------------------------------
  Fx = pF*np.cos(theta)
  Fy = pF*np.sin(theta)
  Fxp = pFp*np.cos(thetap)
  Fyp = pFp*np.sin(thetap)

  oFx = o_feed[1]*np.cos(theta)
  oFy = o_feed[1]*np.sin(theta)
  oFxp = o_feed[2]*np.cos(thetap)
  oFyp = o_feed[2]*np.sin(thetap)

  #Get_t1
  #--------------------------------------------------------------------------------------
  max_Ux = abs( vec[0] )
  max_Uy = abs( vec[1] )
  if max_Ux > max_Uy:
    pAx = pA
    pAy = max_Uy/max_Ux*pA
    t1 = (abs(Fx)-pA*pT)/pA
    mid_t1 = (abs(Fx)-pA*pT-abs(Fxf))/pA
    mid_tb = np.sqrt( (abs(Fx)-abs(Fxf))*pT/pA )
  elif max_Uy > max_Ux:
    pAx = max_Ux/max_Uy*pA
    pAy = pA
    t1 = (abs(Fy)-pA*pT)/pA
    mid_t1 = (abs(Fy)-pA*pT-abs(Fyf))/pA
    mid_tb = np.sqrt( (abs(Fy)-abs(Fyf))*pT/pA )
  else:
    pAx = pA
    pAy = pA
    t1 = (abs(Fx)-pA*pT)/pA
    mid_t1 = (abs(Fx)-pA*pT-abs(Fxf))/pA
    mid_tb = np.sqrt( (abs(Fx)-abs(Fxf))*pT/pA )
  
  max_Uxp = abs( vecp[0] )
  max_Uyp = abs( vecp[1] )
  if max_Uxp > max_Uyp:
    pAxp = pA
    pAyp = max_Uyp/max_Uxp*pA
  elif max_Uyp > max_Uxp:
    pAxp = max_Uxp/max_Uyp*pA
    pAyp = pA
  else:
    pAxp = pA
    pAyp = pA

  if bool_pT == True:
    delta_x = float( abs(pAx/(2*pT)*(delta_t)**2) )
    delta_y = float( abs(pAy/(2*pT)*(delta_t)**2) )
  else:
    delta_x = 0.
    delta_y = 0.

  #Corner Velocity
  #--------------------------------------------------------------------------------------
  # Dx = pD*np.cos(theta)
  # Dy = pD*np.sin(theta)
  # Dxp = pD*np.cos(thetap)
  # Dyp = pD*np.sin(thetap)
  # print("----------------------0")
  # print(abs(pD*np.cos(theta)), abs(Fx)-pAx*pT+pAx/(2*pT)*(delta_t)**2)
  # print(abs(pD*np.sin(theta)), abs(Fy)-pAy*pT+pAy/(2*pT)*(delta_t)**2)
  # print(abs(pD*np.cos(thetap)), abs(Fxp)-pAxp*pT+pAxp/(2*pT)*(delta_t)**2)
  # print(abs(pD*np.sin(thetap)), abs(Fyp)-pAyp*pT+pAyp/(2*pT)*(delta_t)**2)
  # print("-----------------------1")
  Dx = np.min([ abs(pD*np.cos(theta)), abs(oFx)-pAx*pT+pAx/(2*pT)*(delta_t)**2 ])*Xsign
  Dy = np.min([ abs(pD*np.sin(theta)), abs(oFy)-pAy*pT+pAy/(2*pT)*(delta_t)**2 ])*Ysign
  Dxp = np.min([ abs(pD*np.cos(thetap)), abs(oFxp)-pAxp*pT+pAxp/(2*pT)*(delta_t)**2 ])*Xpsign
  Dyp = np.min([ abs(pD*np.sin(thetap)), abs(oFyp)-pAyp*pT+pAyp/(2*pT)*(delta_t)**2 ])*Ypsign

  LDx = Last_D[0]
  LDy = Last_D[1]
  # np.min([pF/Rc, maxV-pA*pT*60/1000])
  if Link[1] == 0:
    Dx=0.
    Dy=0.
    Dxp=0.
    Dyp=0.
    
  if abs(Dx) < 10**(-8):
    Dx=0.
  if abs(Dy) < 10**(-8):
    Dy=0.
  if abs(Dxp) < 10**(-8):
    Dxp=0.
  if abs(Dyp) < 10**(-8):
    Dyp=0.

  #Deviation of Displacement
  #--------------------------------------------------------------------------------------
  if np.cos(theta)*np.sin(thetap)-np.cos(thetap)*np.sin(theta) != 0:
    U_fix = float(( (Dx+Dxp)*delta_t*np.sin(thetap) - (Dy+Dyp)*delta_t*np.cos(thetap) )/( np.cos(theta)*np.sin(thetap)-np.cos(thetap)*np.sin(theta) ))
    U_fixp = float(( -(Dx+Dxp)*delta_t*np.sin(theta) + (Dy+Dyp)*delta_t*np.cos(theta) )/( np.cos(theta)*np.sin(thetap)-np.cos(thetap)*np.sin(theta) ))
  else:
    U_fix = 0.
    U_fixp = 0.
  
  if radius[1] == 0:
    if radius[0] == 0:
      max_Ux = max_Ux - abs(Last_U[0]*np.cos(theta))
      max_Uy = max_Uy - abs(Last_U[0]*np.sin(theta))
    if radius[2] == 0:
      max_Ux = max_Ux - abs(U_fix*np.cos(theta))
      max_Uy = max_Uy - abs(U_fix*np.sin(theta))

  #Get t2
  #--------------------------------------------------------------------------------------
  [Last_D[0],t2x,dfx] = Get_t2( radius, Fx, pAx, pT, Dx, Dxp, delta_t, delta_x, t1, max_Ux, Link, Last_D[0], bool_pT, Fxf, bool_Ff )
  [Last_D[1],t2y,dfy] = Get_t2( radius, Fy, pAy, pT, Dy, Dyp, delta_t, delta_y, t1, max_Uy, Link, Last_D[1], bool_pT, Fyf, bool_Ff )

  #Save Deviation
  #--------------------------------------------------------------------------------------
  Last_U[0] = U_fixp

  #Time Cut of t1 and t2
  #--------------------------------------------------------------------------------------
  tb_cut = 0.
  tc_cut = 0.
  td_cut = 0.
  if max(t2x,t2y) > 0:
    t1_cut_type=0
    bool_t2 = True
    t2 = max(t2x,t2y)
    t1_cut = 0.
  elif t2x < t2y:
    if bool_Ff == 0:
      t1_cut_type=1
      if radius[0] == 0 and radius[2] == 0:
        if  ( abs(Fx)+pAx*pT/2 )**2 - pAx*abs(t2x)*abs(Fx) > 0:
          bool_t2 = False
          t1_cut = float( ( (abs(Fx)+pAx*pT/2) - np.sqrt( ( abs(Fx)+pAx*pT/2 )**2 - pAx*abs(t2x)*abs(Fx) ) )/pAx )
          t1 = t1-t1_cut
          t2 = 0.
        else:
          print("t1_cut error!")
          return
      else:
        if  ( abs(Fx)+pAx*pT/2 )**2 - 2*pAx*abs(t2x)*abs(Fx) > 0:
          bool_t2 = False
          t1_cut = float( ( (abs(Fx)+pAx*pT/2) - np.sqrt( ( abs(Fx)+pAx*pT/2 )**2 - 2*pAx*abs(t2x)*abs(Fx) ) )/pAx )
          t1 = t1-t1_cut
          t2 = 0.
        else:
          print("t1_cut error!")
          return
    else:
      t1_cut_type=12
      if mid_t1 < 0:
        t_0 = np.sqrt( (Fx-Fxf)*pT/pAx )
        if abs(t2x)*abs(Fx) < (Fx**2-Fxf**2)/2/pAx + (abs(Fx)-abs(Fxf))*pT/2 + (abs(Fx)+abs(Fxf))*t_0:
          bool_t2 = False
          t1_cut = 0.
          t1 = t1-t1_cut
          t2 = 0.

          area=abs(t2x)*abs(Fx)
          poly=np.poly1d(np.array([ -pAx/2/pT/pT,
                        2*pAx*t_0/pT/pT + pAx/pT,
                        -3*Fx/pT+2*Fxf/pT-3*pAx*t_0/pT-pAx/2,
                        2*Fx*t_0/pT+pAx*t_0+3*Fx-Fxf,
                        -area ]))
          roots=np.roots(poly)

          for i in range(roots.size):
            if np.imag(roots[i]) == 0.:
              roots[i] = np.real(roots[i])
            else:
              roots[i] = pT
          tb_cut = np.real(min(roots))
          tc_cut = 2*t_0/pT*tb_cut-tb_cut**2/pT

          if tb_cut < 0 or tb_cut>pT:
            print("tb error!")
            return
        else:
          bool_t2 = False
          area = abs(t2x)*abs(Fx) - ((Fx**2-Fxf**2)/2/pAx + (abs(Fx)-abs(Fxf))*pT/2 + (abs(Fx)+abs(Fxf))*t_0)
          t1_cut = 0.
          t1 = t1-t1_cut
          t2 = 0.
          tb_cut = t_0
          tc_cut = t_0**2/pT
          if (abs(Fxf)+pAx*pT/2 )**2 - 2*pAx*area > 0:
            td_cut = float( ( (abs(Fxf)+pAx*pT/2) - np.sqrt( ( abs(Fxf)+pAx*pT/2 )**2 - 2*pAx*area ) )/pAx )
          else:
            print("td_cut error!")
            return
      else:
        if abs(t2x)*abs(Fx) < (abs(Fx)+abs(Fxf)+2*pAx*pT)*(abs(Fx)-abs(Fxf)-pAx*pT)/pA:
          if  ( abs(Fx)+pAx*pT/2 )**2 - pAx*abs(t2x)*abs(Fx) > 0:
            bool_t2 = False
            t1_cut = float( ( (abs(Fx)+pAx*pT/2) - np.sqrt( ( abs(Fx)+pAx*pT/2 )**2 - pAx*abs(t2x)*abs(Fx) ) )/pAx )
            t1 = t1-t1_cut
            t2 = 0.
          else:
            print("t1_cut error!")
            return
        elif abs(t2x)*abs(Fx) - (abs(Fx)+abs(Fxf)+2*pAx*pT)*(abs(Fx)-abs(Fxf)-pAx*pT)/pA < 2*pAx*pT**2+3*abs(Fxf)*pT:
          if  ( abs(Fx)+pAx*pT/2 )**2 - pAx*abs(t2x)*abs(Fx) > 0:
            bool_t2 = False
            t1_cut = float( (Fx-Fxf-pAx*pT)/pAx )
            t1 = t1-t1_cut
            t2 = 0.

            area=abs(t2x)*abs(Fx) - (Fx+Fxf+2*pAx*pT)*(Fx-Fxf-pAx*pT)/pA
            poly=np.poly1d(np.array([ -pAx/2/pT/pT, 3*pAx/pT, (-13*pAx/2-Fxf/pT), (6*pAx*pT+4*Fxf), -area ]))
            roots=np.roots(poly)
            
            for i in range(roots.size):
              if np.imag(roots[i]) == 0.:
                roots[i] = np.real(roots[i])
              else:
                roots[i] = pT
            tb_cut = np.real(min(roots))
            tc_cut = 2*tb_cut-tb_cut**2/pT

            if tb_cut < 0 or tb_cut>pT:
              print("tb error!")
              return
          else:
            print("t1_cut error!")
            return
        else:
          if (abs(Fxf)+pAx*pT/2 )**2 - 2*pAx*area > 0:
            bool_t2 = False
            area = abs(t2x)*abs(Fx) - (abs(Fx)+abs(Fxf)+2*pAx*pT)*(abs(Fx)-abs(Fxf)-pAx*pT)/pA - (2*pAx*pT**2+3*abs(Fxf)*pT)
            t1_cut = float( (Fx-Fxf-pAx*pT)/pAx )
            t1 = t1-t1_cut
            t2 = 0.
            tb_cut = pT
            tc_cut = pT
            td_cut = float( ( (abs(Fxf)+pAx*pT/2) - np.sqrt( ( abs(Fxf)+pAx*pT/2 )**2 - 2*pAx*area ) )/pAx )
          else:
            print("td_cut error!")
            return
  else:
    if bool_Ff == 0:
      if radius[0] == 0 and radius[2] == 0 :
        t1_cut_type=2
        if  ( abs(Fy)+pAy*pT/2 )**2 - pAy*abs(t2y)*abs(Fy) > 0:
          bool_t2 = False
          t1_cut = float( ( (abs(Fy)+pAy*pT/2) - np.sqrt( ( abs(Fy)+pAy*pT/2 )**2 - pAy*abs(t2y)*abs(Fy) ) )/pAy )
          t1 = t1-t1_cut
          t2 = 0.
        else:
          print("t1_cut error!")
          return
      else:
        t1_cut_type=2
        if  ( abs(Fy)+pAy*pT/2 )**2 - 2*pAy*abs(t2y)*abs(Fy) > 0:
          bool_t2 = False
          t1_cut = float( ( (abs(Fy)+pAy*pT/2) - np.sqrt( ( abs(Fy)+pAy*pT/2 )**2 - 2*pAy*abs(t2y)*abs(Fy) ) )/pAy )
          t1 = t1-t1_cut
          t2 = 0.
        else:
          print("t1_cut error!")
          return
    else:
      t1_cut_type=22
      if mid_t1 < 0:
        t_0 = np.sqrt( (Fy-Fyf)*pT/pAy )
        if abs(t2y)*abs(Fy) < (Fy**2-Fyf**2)/2/pAy + (abs(Fy)-abs(Fyf))*pT/2 + (abs(Fy)+abs(Fyf))*t_0:
          bool_t2 = False
          t1_cut = 0.
          t1 = t1-t1_cut
          t2 = 0.

          area=abs(t2y)*abs(Fy)
          poly=np.poly1d(np.array([ -pAy/2/pT/pT,
                        2*pAy*t_0/pT/pT + pAy/pT,
                        -3*Fy/pT+2*Fyf/pT-3*pAy*t_0/pT-pAy/2,
                        2*Fy*t_0/pT+pAy*t_0+3*Fy-Fyf,
                        -area ]))
          roots=np.roots(poly)

          for i in range(roots.size):
            if np.imag(roots[i]) == 0.:
              roots[i] = np.real(roots[i])
            else:
              roots[i] = pT
          tb_cut = np.real(min(roots))
          tc_cut = 2*t_0/pT*tb_cut-tb_cut**2/pT

          if tb_cut < 0 or tb_cut>pT:
            print("tb error!")
            return
        else:
          bool_t2 = False
          area = abs(t2y)*abs(Fy) - ((Fy**2-Fyf**2)/2/pAy + (abs(Fy)-abs(Fyf))*pT/2 + (abs(Fy)+abs(Fyf))*t_0)
          t1_cut = 0.
          t1 = t1-t1_cut
          t2 = 0.
          tb_cut = t_0
          tc_cut = t_0**2/pT
          if (abs(Fyf)+pAy*pT/2 )**2 - 2*pAy*area > 0:
            td_cut = float( ( (abs(Fyf)+pAy*pT/2) - np.sqrt( ( abs(Fyf)+pAy*pT/2 )**2 - 2*pAy*area ) )/pAy )
          else:
            print("td_cut error!")
            return
      else:
        if abs(t2y)*abs(Fy) < (abs(Fy)+abs(Fyf)+2*pAy*pT)*(abs(Fy)-abs(Fyf)-pAy*pT)/pA:
          if  ( abs(Fy)+pAy*pT/2 )**2 - pAy*abs(t2y)*abs(Fy) > 0:
            bool_t2 = False
            t1_cut = float( ( (abs(Fy)+pAy*pT/2) - np.sqrt( ( abs(Fy)+pAy*pT/2 )**2 - pAy*abs(t2y)*abs(Fy) ) )/pAy )
            t1 = t1-t1_cut
            t2 = 0.
          else:
            print("t1_cut error!")
            return
        elif abs(t2y)*abs(Fy) - (abs(Fy)+abs(Fyf)+2*pAy*pT)*(abs(Fy)-abs(Fyf)-pAy*pT)/pA < 2*pAy*pT**2+3*abs(Fyf)*pT:
          if  ( abs(Fy)+pAy*pT/2 )**2 - pAy*abs(t2y)*abs(Fy) > 0:
            bool_t2 = False
            t1_cut = float( (Fy-Fyf-pAy*pT)/pAy )
            t1 = t1-t1_cut
            t2 = 0.

            area=abs(t2y)*abs(Fy) - (Fy+Fyf+2*pAy*pT)*(Fy-Fyf-pAy*pT)/pA
            poly=np.poly1d(np.array([ -pAy/2/pT/pT, 3*pAy/pT, (-13*pAy/2-Fyf/pT), (6*pAy*pT+4*Fyf), -area ]))
            roots=np.roots(poly)
            
            for i in range(roots.size):
              if np.imag(roots[i]) == 0.:
                roots[i] = np.real(roots[i])
              else:
                roots[i] = pT
            tb_cut = np.real(min(roots))
            tc_cut = 2*tb_cut-tb_cut**2/pT

            if tb_cut < 0 or tb_cut>pT:
              print("tb error!")
              return
          else:
            print("t1_cut error!")
            return
        else:
          bool_t2 = False
          area = abs(t2y)*abs(Fy) - (abs(Fy)+abs(Fyf)+2*pAy*pT)*(abs(Fy)-abs(Fyf)-pAy*pT)/pA - (2*pAy*pT**2+3*abs(Fyf)*pT)
          t1_cut = float( (Fy-Fyf-pAy*pT)/pAy )
          t1 = t1-t1_cut
          t2 = 0.
          tb_cut = pT
          tc_cut = pT
          if (abs(Fyf)+pAy*pT/2 )**2 - 2*pAy*area > 0:
            td_cut = float( ( (abs(Fyf)+pAy*pT/2) - np.sqrt( ( abs(Fyf)+pAy*pT/2 )**2 - 2*pAy*area ) )/pAy )
          else:
            print("td_cut error!")
            return

  dfx = np.hstack([dfx,[t1_cut,t1_cut_type]])
  dfy = np.hstack([dfy,[t1_cut,t1_cut_type]])

  #Modulus
  #--------------------------------------------------------------------------------------
  [Fx,Fy,Dx,Dy,LDx,LDy] = np.array([abs(Fx),abs(Fy),abs(Dx),abs(Dy),abs(LDx),abs(LDy)],dtype=float)
  
  #final_t1
  #--------------------------------------------------------------------------------------
  if LDx == 0. and LDy == 0.:
    Lt1 = np.float( t1 )
  elif LDx > LDy:
    Lt1 = np.float( t1 - (LDx - delta_x)/pAx )
  else:
    Lt1 = np.float( t1 - (LDy - delta_y)/pAy )

  if Dx == 0. and Dy == 0.:
    Rt1 = np.float( t1 )
  elif Dx > Dy:
    Rt1 = np.float( t1 - (Dx - delta_x)/pAx )
  else:
    Rt1 = np.float( t1 - (Dy - delta_y)/pAy )

  bool_Lt1 = 1
  bool_Rt1 = 1
  if abs(Lt1) < 10**(-8):
    Lt1 = 0.
    bool_Lt1 = 0
  if abs(Rt1) < 10**(-8):
    Rt1 = 0.
    bool_Rt1 = 0

  if Lt1<0. or Rt1<0.:
    if Lt1<0.:
      print(Ui)
      print((LDx - delta_x)/pAx)
      print((LDy - delta_y)/pAy)
      print(Lt1,Rt1)
      print(t1_cut)
      print("error! Lt1 under zero!")
    if Rt1<0.:
      print(Ui)
      print("t1:",t1)
      print((Dx - delta_x)/pAx)
      print((Dy - delta_y)/pAy)
      print(Lt1,Rt1)
      print(t1_cut)
      print("error! Rt1 under zero!")
    return
  Ltb = 0.
  Rtb = 0.

  #Partition
  #--------------------------------------------------------------------------------------
  if radius[0] == 0:
    if Link[0] != 0: #not first path
      if (bool_pT == True):
        #1
        T_in=T_en
        T_en=T_in + pT-delta_t
        Ts = (T_en-T_in)/(points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = Xsign*( pAx/pT*(t-(T_in-delta_t)) ) if max_Ux>0 else 0.*t
        v1 = Xsign*( pAx/(2*pT)*(t - (T_in-delta_t))**2 + LDx-delta_x ) if max_Ux>0 else 0.*t
        a2 = Ysign*( pAy/pT*(t-(T_in-delta_t)) ) if max_Uy>0 else 0.*t
        v2 = Ysign*( pAy/(2*pT)*(t - (T_in-delta_t))**2 + LDy-delta_y ) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])
      
      if bool_Lt1:
        #2
        T_in=T_en
        T_en=T_in + Lt1 - tc_cut - td_cut
        Ts = (T_en-T_in)/(2*points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = Xsign*( pAx+0.*t ) if max_Ux>0 else 0.*t
        v1 = Xsign*( pAx*(t-T_en)+Fx-pAx*pT/2-t1_cut*pAx - tc_cut*pAx - td_cut*pAx) if max_Ux>0 else 0.*t
        a2 = Ysign*( pAy+0.*t ) if max_Uy>0 else 0.*t
        v2 = Ysign*( pAy*(t-T_en)+Fy-pAy*pT/2-t1_cut*pAy - tc_cut*pAy - td_cut*pAy) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])
    else: # fist path
      if (bool_pT == True):
        #1
        T_in=T_en
        T_en=T_in + pT
        Ts = (T_en-T_in)/(points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = Xsign*( pAx/pT*(t-T_in) ) if max_Ux>0 else 0.*t
        v1 = Xsign*( pAx/(2*pT)*(t - T_in)**2  ) if max_Ux>0 else 0.*t
        a2 = Ysign*( pAy/pT*(t-T_in) ) if max_Uy>0 else 0.*t
        v2 = Ysign*( pAy/(2*pT)*(t - T_in)**2  ) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])

      if bool_Lt1:
        #2
        T_in=T_en
        T_en=T_in + Lt1 - tc_cut
        Ts = (T_en-T_in)/(2*points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)
        
        a1 = Xsign*( pAx+0.*t ) if max_Ux>0 else 0.*t
        v1 = Xsign*( pAx*(t-T_in)+pAx*pT/2 - tc_cut*pAx - td_cut*pAx ) if max_Ux>0 else 0.*t
        a2 = Ysign*( pAy+0.*t ) if max_Uy>0 else 0.*t
        v2 = Ysign*( pAy*(t-T_in)+pAy*pT/2 - tc_cut*pAy - td_cut*pAy ) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if bool_Ff == 1:
      if (bool_P1772 == True):
        if td_cut == 0:
          #3
          T_in=T_en
          T_en=T_in + pT
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx*(t-T_en)/pT ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx*(t-T_en)**2/(2*pT) + Fx - t1_cut*pAx - tc_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy*(t-T_en)/pT ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy*(t-T_en)**2/(2*pT) + Fy - t1_cut*pAy - tc_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
        else:
          #3
          if pAx == pA:
            Ltc = float( -(Fxf-pAx*pT/2-pAx*td_cut) + np.sqrt( (Fxf-pAx*pT/2-pAx*td_cut)**2 + 2*pAx*(Fxf*pT-pAx*pT*pT/6-td_cut*pAx*pT) ) )/pAx
            if Ltc*pAx > pAx*pT/2+pAx*td_cut:
              Ltb = Ltc-pT/2-td_cut
            else:
              Ltb = 0.
          else:
            Ltc = float( -(Fyf-pAy*pT/2-pAy*td_cut) + np.sqrt( (Fyf-pAy*pT/2-pAy*td_cut)**2 + 2*pAy*(Fyf*pT-pAy*pT**2/6-td_cut*pAy*pT) ) )/pAy
            if Ltc*pAy > pAy*pT/2+pAy*td_cut:
              Ltb = Ltc-pT/2-td_cut
            else:
              Ltb = 0.
          #3
          T_in=T_en
          T_en=T_in + Ltc - Ltb
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( pAx+0.*t ) if max_Ux>0 else 0.*t
          v1 = Xsign*( pAx*(t-T_in)+Fxf-pAx*pT/2-td_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( pAy+0.*t ) if max_Uy>0 else 0.*t
          v2 = Ysign*( pAy*(t-T_in)+Fyf-pAy*pT/2-td_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])

          if Ltb != 0.:
            #3+
            T_in=T_en
            T_en=T_in + Ltb + pA*Ltb*Ltb/2/pF
            Ts = (T_en-T_in)/(points)
            t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

            a1 = Xsign*( 0.*t ) if max_Ux>0 else 0.*t
            v1 = Xsign*( Fxf + 0.*t ) if max_Ux>0 else 0.*t
            a2 = Ysign*( 0.*t ) if max_Uy>0 else 0.*t
            v2 = Ysign*( Fyf + 0.*t ) if max_Uy>0 else 0.*t
            Ti = np.hstack([Ti, t])
            Ax = np.hstack([Ax, a1])
            Ay = np.hstack([Ay, a2])
            Vx = np.hstack([Vx, v1])
            Vy = np.hstack([Vy, v2])
    else:
      if radius[2] != 0 and t1_cut != 0:
        if (bool_P1772 == True):
          #3
          if pAx == pA:
            Ltc = float( -(Fx-pAx*pT/2-pAx*t1_cut) + np.sqrt( (Fx-pAx*pT/2-pAx*t1_cut)**2 + 2*pAx*(Fx*pT-pAx*pT*pT/6-t1_cut*pAx*pT) ) )/pAx
            if Ltc*pAx > pAx*pT/2+pAx*t1_cut:
              Ltb = Ltc-pT/2-t1_cut
            else:
              Ltb = 0.
          else:
            Ltc = float( -(Fy-pAy*pT/2-pAy*t1_cut) + np.sqrt( (Fy-pAy*pT/2-pAy*t1_cut)**2 + 2*pAy*(Fy*pT-pAy*pT**2/6-t1_cut*pAy*pT) ) )/pAy
            if Ltc*pAy > pAy*pT/2+pAy*t1_cut:
              Ltb = Ltc-pT/2-t1_cut
            else:
              Ltb = 0.
          #3
          T_in=T_en
          T_en=T_in + Ltc - Ltb
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( pAx+0.*t ) if max_Ux>0 else 0.*t
          v1 = Xsign*( pAx*(t-T_in)+Fx-pAx*pT/2-t1_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( pAy+0.*t ) if max_Uy>0 else 0.*t
          v2 = Ysign*( pAy*(t-T_in)+Fy-pAy*pT/2-t1_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])

          if Ltb != 0.:
            #3+
            T_in=T_en
            T_en=T_in + Ltb + pA*Ltb*Ltb/2/pF
            Ts = (T_en-T_in)/(points)
            t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

            a1 = Xsign*( 0.*t ) if max_Ux>0 else 0.*t
            v1 = Xsign*( Fx + 0.*t ) if max_Ux>0 else 0.*t
            a2 = Ysign*( 0.*t ) if max_Uy>0 else 0.*t
            v2 = Ysign*( Fy + 0.*t ) if max_Uy>0 else 0.*t
            Ti = np.hstack([Ti, t])
            Ax = np.hstack([Ax, a1])
            Ay = np.hstack([Ay, a2])
            Vx = np.hstack([Vx, v1])
            Vy = np.hstack([Vy, v2])
      else:
        if (bool_P1772 == True):
          #3
          T_in=T_en
          T_en=T_in + pT
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx*(t-T_en)/pT ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx*(t-T_en)**2/(2*pT) + Fx - t1_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy*(t-T_en)/pT ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy*(t-T_en)**2/(2*pT) + Fy - t1_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
  
  #circle to line
  #---------------------------------------------------------------------------
  if bool_Ff == -1:
    if mid_t1 > 0:
      if td_cut == 0:
        if (bool_pT == True):
          #7
          T_in=T_en
          T_en=T_in + pT - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( pAx/pT*(t-T_in) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( pAx/(2*pT)*(t - T_in)**2 + abs(Fxf) ) if max_Ux>0 else 0.*t
          a2 = Ysign*( pAy/pT*(t-T_in) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( pAy/(2*pT)*(t - T_in)**2 + abs(Fyf) ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])

        if tb_cut == 0:
          if bool_Rt1:
            #6
            T_in=T_en
            T_en=T_in + mid_t1 - t1_cut
            Ts = (T_en-T_in)/(2*points)
            t = np.arange(T_in+Ts,T_en+Ts/2,Ts)
            
            a1 = Xsign*( pAx+0.*t ) if max_Ux>0 else 0.*t
            v1 = Xsign*( pAx*(t-T_in) + abs(Fxf) + pAx*pT/2 ) if max_Ux>0 else 0.*t
            a2 = Ysign*( pAy+0.*t ) if max_Uy>0 else 0.*t
            v2 = Ysign*( pAy*(t-T_in) + abs(Fyf) + pAy*pT/2 ) if max_Uy>0 else 0.*t
            Ti = np.hstack([Ti, t])
            Ax = np.hstack([Ax, a1])
            Ay = np.hstack([Ay, a2])
            Vx = np.hstack([Vx, v1])
            Vy = np.hstack([Vy, v2])

        if (bool_pT == True):
          #5
          T_in=T_en
          T_en=T_in + pT - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx/pT*(t-T_en) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx/(2*pT)*(t - T_en)**2 + abs(Fxf) + pAx*pT + (mid_t1 - t1_cut - tc_cut)*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy/pT*(t-T_en) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy/(2*pT)*(t - T_en)**2 + abs(Fyf) + pAy*pT + (mid_t1 - t1_cut - tc_cut)*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
    else:
      if td_cut == 0:
        if (bool_pT == True):
          #7
          T_in=T_en
          T_en=T_in + mid_tb - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( pAx/pT*(t-T_in) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( pAx/(2*pT)*(t - T_in)**2 + abs(Fxf) ) if max_Ux>0 else 0.*t
          a2 = Ysign*( pAy/pT*(t-T_in) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( pAy/(2*pT)*(t - T_in)**2 + abs(Fyf) ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])

          #5
          T_in=T_en
          T_en=T_in + mid_tb - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx/pT*(t-T_en) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx/(2*pT)*(t - T_en)**2 + abs(Fxf) + pAx*pT + (mid_t1 - t1_cut - tc_cut)*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy/pT*(t-T_en) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy/(2*pT)*(t - T_en)**2 + abs(Fyf) + pAy*pT + (mid_t1 - t1_cut - tc_cut)*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
  #---------------------------------------------------------------------------

  if (bool_t2 == True) and tb_cut == 0:
    #4
    T_in=T_en
    T_en=T_in + t2
    Ts = (T_en-T_in)/(20*points)
    t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

    a1 = Xsign*( 0.*t ) if max_Ux>0 else 0.*t
    v1 = Xsign*( Fx+0*t - t1_cut*pAx ) if max_Ux>0 else 0.*t
    a2 = Ysign*( 0.*t ) if max_Uy>0 else 0.*t
    v2 = Ysign*( Fy+0*t - t1_cut*pAy ) if max_Uy>0 else 0.*t
    Ti = np.hstack([Ti, t])
    Ax = np.hstack([Ax, a1])
    Ay = np.hstack([Ay, a2])
    Vx = np.hstack([Vx, v1])
    Vy = np.hstack([Vy, v2])
  
  #line to circle
  #---------------------------------------------------------------------------
  if bool_Ff == 1:
    if mid_t1 > 0:
      if td_cut == 0:
        if (bool_pT == True):
          #5
          T_in=T_en
          T_en=T_in + pT - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx/pT*(t-T_in) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx/(2*pT)*(t - T_in)**2 + abs(Fxf) + pAx*pT + (mid_t1 - t1_cut - tc_cut)*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy/pT*(t-T_in) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy/(2*pT)*(t - T_in)**2 + abs(Fyf) + pAy*pT + (mid_t1 - t1_cut - tc_cut)*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])

        if tb_cut == 0:
          if bool_Rt1:
            #6
            T_in=T_en
            T_en=T_in + mid_t1 - t1_cut
            Ts = (T_en-T_in)/(2*points)
            t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

            a1 = Xsign*( -pAx+0.*t ) if max_Ux>0 else 0.*t
            v1 = Xsign*( -pAx*(t-T_en) + abs(Fxf) + pAx*pT/2 ) if max_Ux>0 else 0.*t
            a2 = Ysign*( -pAy+0.*t ) if max_Uy>0 else 0.*t
            v2 = Ysign*( -pAy*(t-T_en) + abs(Fyf) + pAy*pT/2 ) if max_Uy>0 else 0.*t
            Ti = np.hstack([Ti, t])
            Ax = np.hstack([Ax, a1])
            Ay = np.hstack([Ay, a2])
            Vx = np.hstack([Vx, v1])
            Vy = np.hstack([Vy, v2])
          
        if (bool_pT == True):
          #7
          T_in=T_en
          T_en=T_in + pT - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( pAx/pT*(t-T_en) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( pAx/(2*pT)*(t - T_en)**2 + abs(Fxf) ) if max_Ux>0 else 0.*t
          a2 = Ysign*( pAy/pT*(t-T_en) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( pAy/(2*pT)*(t - T_en)**2 + abs(Fyf) ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
    else:
      if td_cut == 0:
        if (bool_pT == True):
          #5
          T_in=T_en
          T_en=T_in + mid_tb - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx/pT*(t-T_in) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx/(2*pT)*(t - T_in)**2 + abs(Fxf) + pAx*pT + (mid_t1 - t1_cut - tc_cut)*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy/pT*(t-T_in) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy/(2*pT)*(t - T_in)**2 + abs(Fyf) + pAy*pT + (mid_t1 - t1_cut - tc_cut)*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])

          #7
          T_in=T_en
          T_en=T_in + mid_tb - tb_cut
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( pAx/pT*(t-T_en) ) if max_Ux>0 else 0.*t
          v1 = Xsign*( pAx/(2*pT)*(t - T_en)**2 + abs(Fxf) ) if max_Ux>0 else 0.*t
          a2 = Ysign*( pAy/pT*(t-T_en) ) if max_Uy>0 else 0.*t
          v2 = Ysign*( pAy/(2*pT)*(t - T_en)**2 + abs(Fyf) ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
  #---------------------------------------------------------------------------
  if radius[2] == 0:
    if bool_Ff == -1:
      if (bool_P1772 == True):
        if td_cut == 0:
          #5
          T_in=T_en
          T_en=T_in + pT
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx*(t-T_in)/pT ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx*(t-T_in)**2/(2*pT) + Fx - t1_cut*pAx - tc_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy*(t-T_in)/pT ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy*(t-T_in)**2/(2*pT) + Fy - t1_cut*pAy - tc_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
        else:
          #5
          if pAx == pA:
            Rtc = float( -(Fxf-pAx*pT/2-pAx*td_cut) + np.sqrt( (Fxf-pAx*pT/2-pAx*td_cut)**2 + 2*pAx*(Fxf*pT-pAx*pT*pT/6-td_cut*pAx*pT) ) )/pAx
            if Rtc*pAx > pAx*pT/2+pAx*td_cut:
              Rtb = Rtc-pT/2-td_cut
            else:
              Rtb = 0.
          else:
            Rtc = float( -(Fyf-pAy*pT/2-pAy*td_cut) + np.sqrt( (Fyf-pAy*pT/2-pAy*td_cut)**2 + 2*pAy*(Fyf*pT-pAy*pT**2/6-td_cut*pAy*pT) ) )/pAy
            if Rtc*pAy > pAy*pT/2+pAy*td_cut:
              Rtb = Rtc-pT/2-td_cut
            else:
              Rtb = 0.

          if Rtb != 0.:
            #5+
            T_in=T_en
            T_en=T_in + Rtb + pA*Rtb*Rtb/2/pF
            Ts = (T_en-T_in)/(points)
            t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

            a1 = Xsign*( 0.*t ) if max_Ux>0 else 0.*t
            v1 = Xsign*( Fxf + 0.*t ) if max_Ux>0 else 0.*t
            a2 = Ysign*( 0.*t ) if max_Uy>0 else 0.*t
            v2 = Ysign*( Fyf + 0.*t ) if max_Uy>0 else 0.*t
            Ti = np.hstack([Ti, t])
            Ax = np.hstack([Ax, a1])
            Ay = np.hstack([Ay, a2])
            Vx = np.hstack([Vx, v1])
            Vy = np.hstack([Vy, v2])

          #5
          T_in=T_en
          T_en=T_in + Rtc - Rtb
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx+0.*t ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx*(t-T_en)+Fxf-pAx*pT/2-td_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy+0.*t ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy*(t-T_en)+Fyf-pAy*pT/2-td_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
    else:
      if radius[0] != 0 and t1_cut != 0:
        if (bool_P1772 == True):
          #5
          if pAx == pA:
            Rtc = float( -(Fx-pAx*pT/2-pAx*t1_cut) + np.sqrt( (Fx-pAx*pT/2-pAx*t1_cut)**2 + 2*pAx*(Fx*pT-pAx*pT*pT/6-t1_cut*pAx*pT) ) )/pAx
            if Rtc*pAx > pAx*pT/2+pAx*t1_cut:
              Rtb = Rtc-pT/2-t1_cut
            else:
              Rtb = 0.
          else:
            Rtc = float( -(Fy-pAy*pT/2-pAy*t1_cut) + np.sqrt( (Fy-pAy*pT/2-pAy*t1_cut)**2 + 2*pAy*(Fy*pT-pAy*pT**2/6-t1_cut*pAy*pT) ) )/pAy
            if Rtc*pAy > pAy*pT/2+pAy*t1_cut:
              Rtb = Rtc-pT/2-t1_cut
            else:
              Rtb = 0.

          if Rtb != 0.:
            #5+
            T_in=T_en
            T_en=T_in + Rtb + pA*Rtb*Rtb/2/pF
            Ts = (T_en-T_in)/(points)
            t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

            a1 = Xsign*( 0.*t ) if max_Ux>0 else 0.*t
            v1 = Xsign*( Fx + 0.*t ) if max_Ux>0 else 0.*t
            a2 = Ysign*( 0.*t ) if max_Uy>0 else 0.*t
            v2 = Ysign*( Fy + 0.*t ) if max_Uy>0 else 0.*t
            Ti = np.hstack([Ti, t])
            Ax = np.hstack([Ax, a1])
            Ay = np.hstack([Ay, a2])
            Vx = np.hstack([Vx, v1])
            Vy = np.hstack([Vy, v2])

          #5
          T_in=T_en
          T_en=T_in + Rtc - Rtb
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx+0.*t ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx*(t-T_en)+Fx-pAx*pT/2-t1_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy+0.*t ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy*(t-T_en)+Fy-pAy*pT/2-t1_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])

      else:
        if (bool_P1772 == True):
          #5
          T_in=T_en
          T_en=T_in + pT
          Ts = (T_en-T_in)/(points)
          t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

          a1 = Xsign*( -pAx*(t-T_in)/pT ) if max_Ux>0 else 0.*t
          v1 = Xsign*( -pAx*(t-T_in)**2/(2*pT) + Fx - t1_cut*pAx ) if max_Ux>0 else 0.*t
          a2 = Ysign*( -pAy*(t-T_in)/pT ) if max_Uy>0 else 0.*t
          v2 = Ysign*( -pAy*(t-T_in)**2/(2*pT) + Fy - t1_cut*pAy ) if max_Uy>0 else 0.*t
          Ti = np.hstack([Ti, t])
          Ax = np.hstack([Ax, a1])
          Ay = np.hstack([Ay, a2])
          Vx = np.hstack([Vx, v1])
          Vy = np.hstack([Vy, v2])
    
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if Link[1] != 0:
      if bool_Rt1:
        #6
        T_in=T_en
        T_en=T_in + Rt1 - tc_cut - td_cut
        Ts = (T_en-T_in)/(2*points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = Xsign*( -pAx+0.*t ) if max_Ux>0 else 0.*t
        v1 = Xsign*( -pAx*(t-T_in)+Fx-pAx*pT/2-t1_cut*pAx-tc_cut*pAx-td_cut*pAx ) if max_Ux>0 else 0.*t
        a2 = Ysign*( -pAy+0.*t ) if max_Uy>0 else 0.*t
        v2 = Ysign*( -pAy*(t-T_in)+Fy-pAy*pT/2-t1_cut*pAy-tc_cut*pAy-td_cut*pAy ) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])
      
      if (bool_pT == True):
        #7
        T_in=T_en
        T_en=T_in + pT-delta_t
        Ts = (T_en-T_in)/(points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = Xsign*( pAx/pT*(t-(T_en+delta_t)) ) if max_Ux>0 else 0.*t
        v1 = Xsign*( pAx/(2*pT)*(t - (T_en+delta_t))**2 + Dx-delta_x ) if max_Ux>0 else 0.*t
        a2 = Ysign*( pAy/pT*(t-(T_en+delta_t)) ) if max_Uy>0 else 0.*t
        v2 = Ysign*( pAy/(2*pT)*(t - (T_en+delta_t))**2 + Dy-delta_y ) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])
    
      if vec[0] < 0.:
        Dx = -Dx
      if vec[1] < 0.:
        Dy = -Dy

      if P1769 == 0:
        #8
        T_in=T_en
        T_en=T_in + 2*delta_t
        Ts = (T_en-T_in)/(points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = 1.4*(Dxp-Dx)/(3*2*delta_t) +0.*t
        v1 = (Dxp-Dx)/(2*delta_t)*(t-T_in) + Dx
        a2 = 1.4*(Dyp-Dy)/(3*2*delta_t) +0.*t
        v2 = (Dyp-Dy)/(2*delta_t)*(t-T_in) + Dy
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])

        # alx = Xsign*( pAx/pT*(-delta_t) )
        # amx = 1.4*(Dxp-Dx)/(3*2*delta_t)
        # # amx = (Dxp-Dx)/(2*delta_t)
        # arx = Xsign*( pAxp/pT*(delta_t) )
        # aly = Ysign*( pAy/pT*(-delta_t) )
        # amy = 1.4*(Dyp-Dy)/(3*2*delta_t)
        # # amy = (Dyp-Dy)/(2*delta_t)
        # ary = Ysign*( pAyp/pT*(delta_t) )

        # #8
        # T_in=T_en
        # T_en=T_in + delta_t
        # Ts = (T_en-T_in)/(points/2)
        # t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        # a1 = (amx-alx)/delta_t*(t-T_in) + alx
        # v1 = (Dxp-Dx)/(2*delta_t)*(t-T_in) + Dx
        # a2 = (amy-aly)/delta_t*(t-T_in) + aly
        # v2 = (Dyp-Dy)/(2*delta_t)*(t-T_in) + Dy
        # Ti = np.hstack([Ti, t])
        # Ax = np.hstack([Ax, a1])
        # Ay = np.hstack([Ay, a2])
        # Vx = np.hstack([Vx, v1])
        # Vy = np.hstack([Vy, v2])

        # #8+
        # T_in=T_en
        # T_en=T_in + delta_t
        # Ts = (T_en-T_in)/(points/2)
        # t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        # a1 = (arx-amx)/delta_t*(t-T_in) + amx
        # v1 = (Dxp-Dx)/(2*delta_t)*(t-T_in+delta_t) + Dx
        # a2 = (ary-amy)/delta_t*(t-T_in) + amy
        # v2 = (Dyp-Dy)/(2*delta_t)*(t-T_in+delta_t) + Dy
        # Ti = np.hstack([Ti, t])
        # Ax = np.hstack([Ax, a1])
        # Ay = np.hstack([Ay, a2])
        # Vx = np.hstack([Vx, v1])
        # Vy = np.hstack([Vy, v2])
      else:
        #8
        T_in=T_en
        T_en=T_in + 2*delta_t
        Ts = (T_en-T_in)/(points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = (Dxp-Dx)/(2*delta_t) +0.*t
        v1 = (Dxp-Dx)/(2*delta_t)*(t-T_in) + Dx
        a2 = (Dyp-Dy)/(2*delta_t) +0.*t
        v2 = (Dyp-Dy)/(2*delta_t)*(t-T_in) + Dy
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])
    else:
      if bool_Rt1:
        #6
        T_in=T_en
        T_en=T_in + Rt1 - tc_cut - td_cut
        Ts = (T_en-T_in)/(2*points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = Xsign*( -pAx+0.*t ) if max_Ux>0 else 0.*t
        v1 = Xsign*( -pAx*(t-T_en-pT/2) ) if max_Ux>0 else 0.*t
        a2 = Ysign*( -pAy+0.*t ) if max_Uy>0 else 0.*t
        v2 = Ysign*( -pAy*(t-T_en-pT/2) ) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])
      
      if (bool_pT == True):
        #7
        T_in=T_en
        T_en=T_in + pT
        Ts = (T_en-T_in)/(points)
        t = np.arange(T_in+Ts,T_en+Ts/2,Ts)

        a1 = Xsign*( pAx/pT*(t-T_en) ) if max_Ux>0 else 0.*t
        v1 = Xsign*( pAx/(2*pT)*(t - T_en)**2 ) if max_Ux>0 else 0.*t
        a2 = Ysign*( pAy/pT*(t-T_en) ) if max_Uy>0 else 0.*t
        v2 = Ysign*( pAy/(2*pT)*(t - T_en)**2 ) if max_Uy>0 else 0.*t
        Ti = np.hstack([Ti, t])
        Ax = np.hstack([Ax, a1])
        Ay = np.hstack([Ay, a2])
        Vx = np.hstack([Vx, v1])
        Vy = np.hstack([Vy, v2])
  
  #--------------------------------------------------------------------------------------
  return [Ti,Ax,Vx,Ay,Vy,Last_U,Last_D,dfx,dfy]

def Get_t2( radius, pF, pA, pT, pD, pDp, delta_t, delta_x, t1, max_U, Link, Last_D, bool_pT, F_f, bool_Ff ):
  #End U
  #--------------------------------------------------------------------------------------
  end_U = float( pA*pT*pT/6 #1 V
          + abs(pF)*t1/2 #2 V
          + abs(pF)*pT - pA*pT*pT/6 #3 V)
          )
  
  #Left U
  #--------------------------------------------------------------------------------------
  if pA < 10**(-8):
    left_U = 0.
  else:
    if bool_pT == True:
      left_U = float( abs(pF)*pT - pA*pT*pT/6 #5 V
              + (abs(pF)-pA*pT/2)*(abs(pF)-pA*pT/2)/(2*pA) #big triangle
              - (abs(Last_D) - delta_x + pA*pT/2)**2/(2*pA) #small triangle
              + pA*(pT**3 - (delta_t)**3)/(6*pT) #arc
              - (pT-delta_t)*delta_x
              + abs(Last_D)*(pT-delta_t) #rectangle
      )
    else:
      left_U = float( abs(pF)*pT - pA*pT*pT/6 #5 X
              + (abs(pF)-pA*pT/2)*(abs(pF)-pA*pT/2)/(2*pA) #big triangle
              - (abs(Last_D))**2/(2*pA) #small triangle
              )

  #Right U
  #--------------------------------------------------------------------------------------
  if pA < 10**(-8):
    link_U = 0.
  else:
    if bool_pT == True:
      link_U = float( abs(pF)*pT - pA*pT*pT/6 #5 V
              + (abs(pF)-delta_x+abs(pD))*(t1-(abs(pD)-delta_x)/pA)/2 #big triangle - small triangle
              + pA*(pT**3 - (delta_t)**3)/(6*pT) #arc
              - delta_x*(pT-delta_t)
              + abs(pD)*(pT-delta_t) #rectangle
      )
    else:
      link_U = float( abs(pF)*pT - pA*pT*pT/6 #5 X
              + (abs(pF)-pA*pT/2)*(abs(pF)-pA*pT/2)/(2*pA) #big triangle
              - (abs(pD))**2/(2*pA) #small triangle
              )

  #mid U for F_final
  #--------------------------------------------------------------------------------------
  if abs(pF)-abs(F_f) < 10**(-8):
    mid_U = 0.
  elif abs(pF)-abs(F_f) > pA*pT:
    mid_U = (abs(pF)+abs(F_f))*(pT+(abs(pF)-abs(F_f)-pA*pT)/(2*pA))
  else:
    mid_U = (abs(pF)+abs(F_f))*np.sqrt((abs(pF)-abs(F_f))*pT/pA)

  #Get t2
  #--------------------------------------------------------------------------------------
  sum_U = 0.
  if radius[0] == 0:
    sum_U = sum_U + float( Link[0]*(left_U) + (1-Link[0])*end_U ) + mid_U
  if radius[2] == 0:
    sum_U = sum_U + float( Link[1]*(link_U) + (1-Link[1])*end_U ) + mid_U
  
  if abs(pF) < 10**(-8):
    t2 = 0.
  else:
    t2 = float( (max_U-sum_U)/abs(pF) )
  
  #DataFrame
  #--------------------------------------------------------------------------------------
  df = np.array( [pF, pD, pDp, end_U, link_U, left_U, Last_D, sum_U, max_U, t1, t2] )
  return pDp,t2,df

def calcir(Ui, radius, feed, parms, Ti, Ax, Vx, Ay, Vy, eVx, eVy, eAx, eAy):
  # print(f"L: {abs(Ax[-1]) ,abs(Ay[-1])}, R: {abs(eAx[1]) ,abs(eAy[1]) }")
  # if radius[2] != 0 and radius[0] == 0:
  #   print(f"start: {abs(Ax[-1]) != 0 or abs(Ay[-1]) != 0}")
  # elif radius[2] == 0 and radius[0] != 0:
  #   print(f"end: {abs(eAx[1]) != 0 or abs(eAy[1]) != 0}")
  
  #Input Data; parms = [F, P1660, P1772, P1783]
  #--------------------------------------------------------------------------------------
  pF = feed[1]
  pA = parms[0] # P1660
  pT = parms[1] # P1772
  pD = parms[2] # P1783
  ac_sup = parms[3] # P1735
  vc_inf = parms[4] # P1732
  vec = np.array( [ Ui[1][0] - Ui[0][0] , Ui[1][1] - Ui[0][1] ], dtype=float)
  theta = np.arctan2( vec[1], vec[0] )
  # points = 100 # Number of points
  T_en = Ti[-1] # Beginning Time of each segment
  
  if np.sqrt(abs(radius[1])*ac_sup) < vc_inf and vc_inf < pF:
    pF = vc_inf
  elif vc_inf < np.sqrt(abs(radius[1])*ac_sup) and np.sqrt(abs(radius[1])*ac_sup) < pF:
    pF = np.sqrt(abs(radius[1])*ac_sup)

  if radius[1] > 0:
    CW = 1.
  else:
    CW = -1.
  
  if (vec[0]**2+vec[1]**2)/4 > radius[1]**2:
    MO = 0.
    print("radius error!")
  else:
    MO = radius[1]/abs(radius[1])*np.sqrt(radius[1]**2-(vec[0]**2+vec[1]**2)/4)
  centre = np.array([ (Ui[1][0]+Ui[0][0])/2 - MO*np.sin(theta) , (Ui[1][1]+Ui[0][1])/2 + MO*np.cos(theta) ], dtype=float)
  Ctheta = np.arctan2( Ui[0][1]-centre[1], Ui[0][0]-centre[0] ) #Initial angle
  r = abs(radius[1])
  
  if radius[2] != 0 and radius[0] == 0:
    LF = np.sqrt(Vx[-1]**2 + Vy[-1]**2)
    if LF == pF:
      Tcircle = 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r )/pF
    elif abs(Ax[-1]) == 0 and abs(Ay[-1]) == 0:
      if pF-LF < pA*pT:
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + ( pF-LF )*np.sqrt( (pF-LF)*pT/pA ) )/pF
      else:
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + ( (pF-LF)/pA+pT )*( pF-LF )/2 )/pF
    else:
      if pF-LF < pA*pT/2:#####################################
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + pA*pT**2/6 + (pF-LF+pA*pT/2)*(pF-LF-pA*pT/2)/pA/2 )/pF
      else:
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + pA*pT**2/6 + (pF-LF+pA*pT/2)*(pF-LF-pA*pT/2)/pA/2 )/pF
    
    T_in = T_en
    T_en = T_in + Tcircle
    Ts = (T_en-T_in)/(10*points)
    t = np.arange(T_in+Ts,T_en+Ts/2,Ts)
    th = np.zeros(t.size,dtype=float)
    w = np.zeros(t.size,dtype=float)
    dwdt = np.zeros(t.size,dtype=float)
    
    if LF == pF:
      th = pF/r*(t-T_in)
      w = pF/r + 0.*t
      dwdt = 0. + 0.*t
    elif abs(Ax[-1]) == 0 and abs(Ay[-1]) == 0:
      if pF-LF < pA*pT:
        for i in range(t.size):
          t_b = np.sqrt( (pF-LF)*pT/pA )
          if t[i] < T_in + t_b:
            tt = t[i]-T_in
            th[i] = ( pA/6/pT*tt**3 + LF*tt )/r
            w[i] = ( pA/2/pT*tt**2 + LF )/r
            dwdt[i] = pA/pT*tt/r
          elif t[i] < T_in + 2*t_b:
            tt = t[i] - (T_in + 2*t_b)
            th[i] = (-pA/6/pT*tt**3 + pF*tt )/r - (-pA/6/pT*(-t_b)**3 + pF*(-t_b) )/r + \
              ( pA/6/pT*t_b**3 + LF*t_b )/r
            w[i] = ( -pA/2/pT*tt**2 + pF)/r
            dwdt[i] = -pA/pT*tt/r
          else:
            tt = t[i] - (T_in + 2*t_b)
            th[i] = pF/r*tt - (-pA/6/pT*(-t_b)**3 + pF*(-t_b) )/r + \
              ( pA/6/pT*t_b**3 + LF*t_b )/r
            w[i] = pF/r
            dwdt[i] = 0.
      else:
        for i in range(t.size):
          if t[i] < T_in + pT:
            tt = t[i]-T_in
            th[i] = ( pA/6/pT*tt**3 + LF*tt )/r 
            w[i] = ( pA/2/pT*tt**2 + LF )/r
            dwdt[i] = pA/pT*tt/r
          elif t[i] < T_in + (pF-LF)/pA:
            tt = t[i] - (T_in+pT)
            th[i] = ( pA/2*tt**2 + (LF+pA*pT/2)*tt )/r + (pA/6/pT*pT**3 + LF*pT )/r 
            w[i] = ( pA*tt + LF + pA*pT/2 )/r
            dwdt[i] = pA/r
          elif t[i] < T_in + (pF-LF)/pA + pT:
            tt = t[i] - (T_in + (pF-LF)/pA + pT)
            th[i] = (-pA/6/pT*tt**3 + pF*tt )/r - (-pA/6/pT*(-pT)**3 + pF*(-pT) )/r + \
              ( pA/2*((pF-LF)/pA-pT)**2 + (LF+pA*pT/2)*((pF-LF)/pA-pT) )/r + (pA/6/pT*pT**3 + LF*pT )/r 
            w[i] = ( -pA/2/pT*tt**2 + pF)/r
            dwdt[i] = -pA/pT*tt/r
          else:
            tt = t[i] - (T_in + (pF-LF)/pA + pT)
            th[i] = pF/r*tt - (-pA/6/pT*(-pT)**3 + pF*(-pT) )/r + \
              ( pA/2*((pF-LF)/pA-pT)**2 + (LF+pA*pT/2)*((pF-LF)/pA-pT) )/r + (pA/6/pT*pT**3 + LF*pT )/r 
            w[i] = pF/r
            dwdt[i] = 0.
    else:
      if pF-LF < pA*pT/2:################################
        for i in range(t.size):
          if t[i] < T_in + (pF-LF)/pA:
            tt = t[i]-T_in
            th[i] = ( pA/2*tt**2 + LF*tt )/r
            w[i] = ( pA*tt + LF )/r
            dwdt[i] = pA/r
          else:
            tt = t[i]-T_in-(pF-LF)/pA
            th[i] = pF/r*tt + ( pA/2*((pF-LF)/pA)**2 + LF*(pF-LF)/pA )/r
            w[i] = pF/r
            dwdt[i] = 0.
      else:
        for i in range(t.size):
          if t[i] < T_in + (pF-LF)/pA-pT/2:
            tt = t[i]-T_in
            th[i] = ( pA/2*tt**2 + LF*tt )/r
            w[i] = ( pA*tt + LF )/r
            dwdt[i] = pA/r
          elif t[i] < T_in + (pF-LF)/pA+pT/2:
            tt = t[i]-(T_in + (pF-LF)/pA+pT/2)
            th[i] = (-pA/6/pT*tt**3 + pF*tt )/r - (-pA/6/pT*(-pT)**3 + pF*(-pT) )/r + \
            ( pA/2*((pF-LF)/pA-pT/2)**2 + LF*((pF-LF)/pA-pT/2) )/r
            w[i] = ( -pA/2/pT*tt**2 + pF)/r
            dwdt[i] = -pA/pT*tt/r
          else:
            tt = t[i] - (T_in + (pF-LF)/pA+pT/2)
            th[i] = pF/r*tt - (-pA/6/pT*(-pT)**3 + pF*(-pT) )/r + ( pA/2*((pF-LF)/pA-pT/2)**2 + LF*((pF-LF)/pA-pT/2) )/r
            w[i] = pF/r
            dwdt[i] = 0.

    v1 = CW*( -r*w*np.sin( Ctheta + CW*th ) )
    v2 = CW*( r*w*np.cos( Ctheta + CW*th ) )
    a1 = -CW*r*dwdt*np.sin( Ctheta + CW*th ) - r*w*w*np.cos( Ctheta + CW*th )
    a2 = CW*r*dwdt*np.cos( Ctheta + CW*th ) - r*w*w*np.sin( Ctheta + CW*th )

  elif radius[2] == 0 and radius[0] != 0:
    RF = np.sqrt(eVx[1]**2 + eVy[1]**2)
    if RF == pF:
      Tcircle = 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r )/pF
    elif abs(eAx[1]) == 0 and abs(eAy[1]) == 0:
      if pF-RF < pA*pT:
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + ( pF-RF )*np.sqrt( (pF-RF)*pT/pA ) )/pF
      else:
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + ( (pF-RF)/pA+pT )*( pF-RF )/2 )/pF
    else:
      if pF-RF < pA*pT/2:#####################################
        # Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + pA*pT**2/6 + (pF-RF+pA*pT/2)*(pF-RF-pA*pT/2)/pA/2 )/pF
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + (pF-RF)**2/2/pA )/pF
      else:
        Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + pA*pT**2/6 + (pF-RF+pA*pT/2)*(pF-RF-pA*pT/2)/pA/2 )/pF
    
    T_in = T_en
    T_en = T_in + Tcircle
    Ts = (T_en-T_in)/(10*points)
    t = np.arange(T_in+Ts,T_en+Ts/2,Ts)
    th = np.zeros(t.size,dtype=float)
    w = np.zeros(t.size,dtype=float)
    dwdt = np.zeros(t.size,dtype=float)
    
    if RF == pF:
      th = pF/r*(t-T_in)
      w = pF/r + 0.*t
      dwdt = 0. + 0.*t
    elif abs(eAx[1]) == 0 and abs(eAy[1]) == 0:
      if pF-RF < pA*pT:
        for i in range(t.size):
          t_b = np.sqrt( (pF-RF)*pT/pA )
          if t[i] < T_in + Tcircle - 2*t_b:
            tt = t[i] - (T_in)
            th[i] = pF/r*tt
            w[i] = pF/r
            dwdt[i] = 0.
          elif t[i] < T_in + Tcircle - t_b:
            tt = t[i] - ( T_in + Tcircle - 2*t_b )
            th[i] = ( -pA/6/pT*tt**3 + pF*tt )/r + pF/r*(Tcircle - 2*t_b)
            w[i] = ( -pA/2/pT*tt**2 + pF)/r
            dwdt[i] = -pA/pT*tt/r
          else:
            tt = t[i] - ( T_in + Tcircle )
            th[i] = ( pA/6/pT*tt**3 + RF*tt )/r - ( pA/6/pT*(-t_b)**3 + RF*(-t_b) )/r + \
              ( -pA/6/pT*t_b**3 + pF*t_b )/r + pF/r*(Tcircle - 2*t_b)
            w[i] = ( pA/2/pT*tt**2 + RF )/r
            dwdt[i] = pA/pT*tt/r
      else:
        for i in range(t.size):
          if t[i] < T_in + Tcircle - ( (pF-RF)/pA + pT ):
            tt = t[i] - (T_in)
            th[i] = pF/r*tt
            w[i] = pF/r
            dwdt[i] = 0.
          elif t[i] < T_in + Tcircle - ( (pF-RF)/pA ):
            tt = t[i] - ( T_in + Tcircle - ( (pF-RF)/pA + pT ))
            th[i] = ( -pA/6/pT*tt**3 + pF*tt )/r + pF/r*( Tcircle - ( (pF-RF)/pA + pT ) )
            w[i] = ( -pA/2/pT*tt**2 + pF)/r
            dwdt[i] = -pA/pT*tt/r
          elif t[i] < T_in + Tcircle - pT:
            tt = t[i] - ( T_in + Tcircle - pT )

            th[i] = ( -pA/2*tt**2 + (RF+pA*pT/2)*tt )/r - \
              ( -pA/2*( pT-(pF-RF)/pA )**2 + (RF+pA*pT/2)*( pT-(pF-RF)/pA ) )/r + \
              ( -pA/6/pT*pT**3 + pF*pT )/r + \
              pF/r*( Tcircle - ( (pF-RF)/pA + pT ) )

            w[i] = ( -pA*tt + RF + pA*pT/2 )/r
            dwdt[i] = -pA/r
          else:
            tt = t[i]- (T_in + Tcircle)

            th[i] = ( pA/6/pT*tt**3 + RF*tt )/r - ( pA/6/pT*(-pT)**3 + RF*(-pT) )/r - \
              ( -pA/2*( pT-(pF-RF)/pA )**2 + (RF+pA*pT/2)*( pT-(pF-RF)/pA ) )/r + \
              ( -pA/6/pT*pT**3 + pF*pT )/r + \
              pF/r*( Tcircle - ( (pF-RF)/pA + pT ) )

            w[i] = ( pA/2/pT*tt**2 + RF )/r
            dwdt[i] = pA/pT*tt/r
    else:
      if pF-RF < pA*pT/2:
        for i in range(t.size):####################################################
          if t[i] < T_in + Tcircle - (pF-RF)/pA:
            tt = t[i]-T_in
            th[i] = pF/r*tt
            w[i] = pF/r
            dwdt[i] = 0.
          else:
            tt = t[i]-(T_in + Tcircle - (pF-RF)/pA)
            th[i] = ( -pA/2*tt**2 + pF*tt )/r + pF/r*(Tcircle - (pF-RF)/pA)
            w[i] = ( -pA*tt + pF )/r
            dwdt[i] = -pA/r
      else:
        for i in range(t.size):
          if t[i] < T_in + Tcircle - ((pF-RF)/pA+pT/2):
            tt = t[i] - T_in
            th[i] = pF/r*tt
            w[i] = pF/r
            dwdt[i] = 0.
          elif t[i] < T_in + Tcircle - ((pF-RF)/pA-pT/2):
            tt = t[i] - (T_in + Tcircle - ((pF-RF)/pA+pT/2))
            th[i] = (-pA/6/pT*tt**3 + pF*tt )/r + pF/r*(Tcircle - ((pF-RF)/pA+pT/2))
            w[i] = ( -pA/2/pT*tt**2 + pF)/r
            dwdt[i] = -pA/pT*tt/r
          else:
            tt = t[i] - ( T_in + Tcircle )
            th[i] = ( -pA/2*tt**2 + RF*tt )/r + 2*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r )
            w[i] = ( -pA*tt + RF )/r
            dwdt[i] = -pA/r

    v1 = CW*( -r*w*np.sin( Ctheta + CW*th ) )
    v2 = CW*( r*w*np.cos( Ctheta + CW*th ) )
    a1 = -CW*r*dwdt*np.sin( Ctheta + CW*th ) - r*w*w*np.cos( Ctheta + CW*th )
    a2 = CW*r*dwdt*np.cos( Ctheta + CW*th ) - r*w*w*np.sin( Ctheta + CW*th )
  else:
    Tcircle = 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r )/pF

    T_in = T_en
    T_en = T_in + Tcircle
    Ts = (T_en-T_in)/(10*points)
    t = np.arange(T_in+Ts,T_en+Ts/2,Ts)
    th = np.zeros(t.size,dtype=float)
    w = np.zeros(t.size,dtype=float)
    dwdt = np.zeros(t.size,dtype=float)
    th = pF/r*(t-T_in)
    w = pF/r + 0.*t
    dwdt = 0. + 0.*t
    # LF = np.sqrt(Vx[-1]**2 + Vy[-1]**2)
    # RF = np.sqrt(eVx[1]**2 + eVy[1]**2)

    # if RF == pF:
    #   Tcircle = 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r )/pF
    # elif pF-RF < pA*pT/2:
    #   Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + 2*( (pF-RF)**2/2/pA ) )/pF
    # else:
    #   Tcircle = ( 2*r*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r ) + 2*( pA*pT**2/6 + (pF-RF+pA*pT/2)*(pF-RF-pA*pT/2)/pA/2 ) )/pF
    
    # T_in = T_en
    # T_en = T_in + Tcircle
    # Ts = (T_en-T_in)/(10*points)
    # t = np.arange(T_in+Ts,T_en+Ts/2,Ts)
    # th = np.zeros(t.size,dtype=float)
    # w = np.zeros(t.size,dtype=float)
    # dwdt = np.zeros(t.size,dtype=float)
    
    # if RF == pF:
    #   th = pF/r*(t-T_in)
    #   w = pF/r + 0.*t
    #   dwdt = 0. + 0.*t
    # elif pF-RF < pA*pT/2:
    #   for i in range(t.size):
    #     if t[i] < T_in + (pF-LF)/pA:###
    #       tt = t[i]-T_in
    #       th[i] = ( pA/2*tt**2 + LF*tt )/r
    #       w[i] = ( pA*tt + LF )/r
    #       dwdt[i] = pA/r
    #     elif t[i] < T_in + Tcircle - (pF-RF)/pA:###
    #       tt = t[i]-T_in
    #       th[i] = pF/r*tt
    #       w[i] = pF/r
    #       dwdt[i] = 0.
    #     else:
    #       tt = t[i]-(T_in + Tcircle - (pF-RF)/pA)
    #       th[i] = ( -pA/2*tt**2 + pF*tt )/r + pF/r*(Tcircle - (pF-RF)/pA)
    #       w[i] = ( -pA*tt + pF )/r
    #       dwdt[i] = -pA/r
    # else:
    #   for i in range(t.size):
    #     if t[i] < T_in + (pF-LF)/pA-pT/2:####
    #       tt = t[i]-T_in
    #       th[i] = ( pA/2*tt**2 + LF*tt )/r
    #       w[i] = ( pA*tt + LF )/r
    #       dwdt[i] = pA/r
    #     elif t[i] < T_in + (pF-LF)/pA+pT/2:####
    #       tt = t[i]-(T_in + (pF-LF)/pA+pT/2)
    #       th[i] = (-pA/6/pT*tt**3 + pF*tt )/r - (-pA/6/pT*(-pT)**3 + pF*(-pT) )/r + \
    #       ( pA/2*((pF-LF)/pA-pT/2)**2 + LF*((pF-LF)/pA-pT/2) )/r
    #       w[i] = ( -pA/2/pT*tt**2 + pF)/r
    #       dwdt[i] = -pA/pT*tt/r
    #     elif t[i] < T_in + Tcircle - ((pF-RF)/pA+pT/2):####
    #       tt = t[i] - T_in
    #       th[i] = pF/r*tt
    #       w[i] = pF/r
    #       dwdt[i] = 0.
    #     elif t[i] < T_in + Tcircle - ((pF-RF)/pA-pT/2):
    #       tt = t[i] - (T_in + Tcircle - ((pF-RF)/pA+pT/2))
    #       th[i] = (-pA/6/pT*tt**3 + pF*tt )/r + pF/r*(Tcircle - ((pF-RF)/pA+pT/2))
    #       w[i] = ( -pA/2/pT*tt**2 + pF)/r
    #       dwdt[i] = -pA/pT*tt/r
    #     else:
    #       tt = t[i] - ( T_in + Tcircle )
    #       th[i] = ( -pA/2*tt**2 + RF*tt )/r + 2*np.arcsin( np.sqrt(vec[0]**2+vec[1]**2)/2/r )
    #       w[i] = ( -pA*tt + RF )/r
    #       dwdt[i] = -pA/r

    v1 = CW*( -r*w*np.sin( Ctheta + CW*th ) )
    v2 = CW*( r*w*np.cos( Ctheta + CW*th ) )
    a1 = -CW*r*dwdt*np.sin( Ctheta + CW*th ) - r*w*w*np.cos( Ctheta + CW*th )
    a2 = CW*r*dwdt*np.cos( Ctheta + CW*th ) - r*w*w*np.sin( Ctheta + CW*th )

  
  Ti = np.hstack([Ti, t])
  Ax = np.hstack([Ax, a1])
  Ay = np.hstack([Ay, a2])
  Vx = np.hstack([Vx, v1])
  Vy = np.hstack([Vy, v2])

  return [Ti,Ax,Vx,Ay,Vy]

def AutoVA(DATA,P1660,P1772,P1783,P1735,P1732,P1737,P1738):
  #P1737&P1738
  #--------------------------------------------------------------------------------------
  if P1737 == 0: P1737 = 9999999
  o_feed = np.hstack([DATA[0,3]/60,DATA[:,3]/60])
  if(DATA.shape[0]>2):
    for i in range(DATA.shape[0]-2):
      if( DATA[i,2] == 0 and DATA[i+1,2] == 0 ):
        path1 = np.sqrt( (DATA[i+1,0]-DATA[i,0])**2 + (DATA[i+1,1]-DATA[i,1])**2 )
        path2 = np.sqrt( (DATA[i+2,0]-DATA[i+1,0])**2 + (DATA[i+2,1]-DATA[i+1,1])**2 )
        theta1 = np.arctan2( DATA[i+1,1]-DATA[i,1], DATA[i+1,0]-DATA[i,0] )
        theta2 = np.arctan2( DATA[i+2,1]-DATA[i+1,1], DATA[i+2,0]-DATA[i+1,0] )
        vel1 = DATA[i,3]/60
        vel2 = DATA[i+1,3]/60
        t_avg = (path1/vel1+path2/vel2)/2
        radio1 = np.sqrt( abs(vel1*np.cos(theta1)-vel2*np.cos(theta2))/t_avg/P1737 )
        radio2 = np.sqrt( abs(vel1*np.sin(theta1)-vel2*np.sin(theta2))/t_avg/P1737 )
        DATA[i,3] = DATA[i,3]/max(radio1,radio2,1.0)
        DATA[i+1,3] = DATA[i+1,3]/max(radio1,radio2,1.0)
        if(DATA[i,3]<P1738): DATA[i,3]=P1738
        if(DATA[i+1,3]<P1738): DATA[i+1,3]=P1738
  # for i in range(DATA.shape[0]-1):
  #   print(f"Path {i+1}: {round(DATA[i,3],5)} (mm/min)")

  #Input Data and Normalize
  #--------------------------------------------------------------------------------------
  feed = np.hstack([DATA[0,3]/60,DATA[:,3]/60])
  P1660=P1660
  P1772=P1772/1000.
  P1783=P1783/60.
  if P1735 == 0: P1735 = 9999999
  P1732=P1732/60.

  # if np.sqrt(np.max(abs(DATA[:,2]))*P1735) - 1.5*P1660*P1772 < P1783:
  #   P1783 = np.sqrt(np.max(abs(DATA[:,2]))*P1735) - 1.5*P1660*P1772

  parms = np.array( [P1660, P1772, P1783, P1735, P1732], dtype=float )
  
  #Initial
  #--------------------------------------------------------------------------------------
  tot_Ti = np.zeros(1, dtype=float)
  tot_Ax = np.zeros(1, dtype=float)
  tot_Vx = np.zeros(1, dtype=float)
  tot_Ay = np.zeros(1, dtype=float)
  tot_Vy = np.zeros(1, dtype=float)
  tot_Timer = np.array(0)

  Link = np.array([0, 1])
  Last_U = np.array([0.,0.])
  Last_D = np.array([0.,0.])
  xDF = np.zeros(13, dtype=float)
  yDF = np.zeros(13, dtype=float)

  eVx = np.zeros(1, dtype=float)
  eVy = np.zeros(1, dtype=float)

  #Iteration
  #--------------------------------------------------------------------------------------
  for i in range(DATA.shape[0]-1):
    #Path with 3 Points
    # if i == 4: break
    #--------------------------------------------------------------------------------------
    if i != DATA.shape[0]-2:
      Ui = DATA[i:i+3,0:2]
      if i == 0:
        radius = np.hstack([ 0., DATA[i:i+2,2] ])
      else:
        radius = DATA[i-1:i+2,2]
    else:
      Ui = np.vstack([ DATA[i:i+2,0:2], [0.,0.] ])
      if DATA.shape[0] == 2:
        radius = np.array([ 0., DATA[i,2], 0. ])
      else:
        radius = np.hstack([ DATA[i-1:i+1,2], 0. ])

    #Link of Final Path
    #--------------------------------------------------------------------------------------
    if i == DATA.shape[0]-2:
      Link[1] = 0

    #AutoVA
    #--------------------------------------------------------------------------------------
    Ti = np.array([tot_Ti[-1]])
    Ax = np.array([tot_Ax[-1]])
    Vx = np.array([tot_Vx[-1]])
    Ay = np.array([tot_Ay[-1]])
    Vy = np.array([tot_Vy[-1]])
    # print(f"{i}: {radius}")
    if radius[1] == 0:
      # print("11111111111111111111111111111111111111")
      [Ti,Ax,Vx,Ay,Vy,Last_U,Last_D,dfx,dfy] = cal(Ui, radius, feed[i:i+3], o_feed[i:i+3], parms, Link, Ti, Ax, Vx, Ay, Vy, Last_U, Last_D)
    else:    
      if radius[2] == 0 and i != DATA.shape[0]-2 :
        eLink = np.array([1, 1])
        j=i+1
        if j != DATA.shape[0]-2:
          eUi = DATA[j:j+3,0:2]
          if j == 0:
            eradius = np.hstack([ 0., DATA[j:j+2,2] ])
          else:
            eradius = DATA[j-1:j+2,2]
        else:
          eUi = DATA[j:j+2,0:2]
          eUi = np.vstack([ DATA[j:j+2,0:2], [0.,0.] ])
          eradius = np.hstack([ DATA[j-1:j+1,2], 0. ])
        if j == DATA.shape[0]-2:
          eLink[1] = 0
        eTi = np.zeros(1, dtype=float)
        eAx = np.zeros(1, dtype=float)
        eVx = np.zeros(1, dtype=float)
        eAy = np.zeros(1, dtype=float)
        eVy = np.zeros(1, dtype=float)
        [eTi,eAx,eVx,eAy,eVy,Last_U,Last_D,dfx,dfy] = cal(eUi, eradius, feed[j:j+3], o_feed[j:j+3], parms, eLink, eTi, eAx, eVx, eAy, eVy, np.array([0.,0.]), np.array([0.,0.]))
      else:
        eVx = np.array([0.,0.,0.,0.], dtype=float)
        eVy = np.array([0.,0.,0.,0.], dtype=float)
        eAx = np.array([0.,0.,0.,0.], dtype=float)
        eAy = np.array([0.,0.,0.,0.], dtype=float)

      [Ti,Ax,Vx,Ay,Vy] = calcir(Ui, radius, feed[i:i+3], parms, Ti, Ax, Vx, Ay, Vy, eVx, eVy, eAx, eAy)
      Last_U = np.array([0.,0.])
      Last_D = np.array([0.,0.])

    # xDF = np.vstack([xDF,dfx])
    # yDF = np.vstack([yDF,dfy])
    tot_Ti = np.hstack([tot_Ti, Ti[1:] ])
    tot_Ax = np.hstack([tot_Ax, Ax[1:] ])
    tot_Vx = np.hstack([tot_Vx, Vx[1:] ])
    tot_Ay = np.hstack([tot_Ay, Ay[1:] ])
    tot_Vy = np.hstack([tot_Vy, Vy[1:] ])
    tot_Timer = np.hstack([tot_Timer, tot_Ti.size])

    #Link of First Path
    #--------------------------------------------------------------------------------------
    if i == 0:
      Link[0] = 1

  #Endpoint
  #--------------------------------------------------------------------------------------
  tot_Ti = np.hstack([tot_Ti, tot_Ti[-1] ])
  tot_Ax = np.hstack([tot_Ax, tot_Ax[-1] ])
  tot_Vx = np.hstack([tot_Vx, tot_Vx[-1] ])
  tot_Ay = np.hstack([tot_Ay, tot_Ay[-1] ])
  tot_Vy = np.hstack([tot_Vy, tot_Vy[-1] ])

  #--------------------------------------------------------------------------------------
  return [1000*tot_Ti,tot_Ax,tot_Vx,tot_Ay,tot_Vy,xDF,yDF,tot_Timer]

def AfterInterpolation(tot_Ti,tot_Vx,tot_Vy,tot_Ax,tot_Ay,P1769,resolution=1):
  new_Ti = np.arange( int( np.ceil(resolution*tot_Ti[-1])+1+resolution*P1769 ) )/resolution
  new_Vx = np.zeros( int(np.ceil(resolution*tot_Ti[-1])+1), dtype=float )
  new_Vy = np.zeros( int(np.ceil(resolution*tot_Ti[-1])+1), dtype=float )
  new_Ax = np.zeros( int(np.ceil(resolution*tot_Ti[-1])+1), dtype=float )
  new_Ay = np.zeros( int(np.ceil(resolution*tot_Ti[-1])+1), dtype=float )

  #Interpolation for 1 ms
  #--------------------------------------------------------------------------------------
  n = 1
  idx = [[0,0]]
  for i in range(tot_Ti.size):
    if n/resolution == tot_Ti[i]:
      idx += [[i,i]]
      n+=1
    elif n/resolution < tot_Ti[i]:
      idx += [[i-1,i]]
      n+=1
  idx = np.array(idx)

  for i in range( int(np.ceil(resolution*tot_Ti[-1])) ):
    if idx[i,0] == idx[i,1]:
      new_Vx[i] = tot_Vx[idx[i,0]]
      new_Vy[i] = tot_Vy[idx[i,0]]
      new_Ax[i] = tot_Ax[idx[i,0]]
      new_Ay[i] = tot_Ay[idx[i,0]]
    else:
      new_Vx[i] = tot_Vx[idx[i,0]] + ( i/resolution-tot_Ti[idx[i,0]] )*( tot_Vx[idx[i,1]] - tot_Vx[idx[i,0]] )/( tot_Ti[idx[i,1]] - tot_Ti[idx[i,0]] )
      new_Vy[i] = tot_Vy[idx[i,0]] + ( i/resolution-tot_Ti[idx[i,0]] )*( tot_Vy[idx[i,1]] - tot_Vy[idx[i,0]] )/( tot_Ti[idx[i,1]] - tot_Ti[idx[i,0]] )
      new_Ax[i] = tot_Ax[idx[i,0]] + ( i/resolution-tot_Ti[idx[i,0]] )*( tot_Ax[idx[i,1]] - tot_Ax[idx[i,0]] )/( tot_Ti[idx[i,1]] - tot_Ti[idx[i,0]] )
      new_Ay[i] = tot_Ay[idx[i,0]] + ( i/resolution-tot_Ti[idx[i,0]] )*( tot_Ay[idx[i,1]] - tot_Ay[idx[i,0]] )/( tot_Ti[idx[i,1]] - tot_Ti[idx[i,0]] )

  #Convolution
  #--------------------------------------------------------------------------------------
  con = np.ones(resolution*P1769+1)/(resolution*P1769+1)
  tot_Ti = np.array(new_Ti)
  tot_Vx = np.convolve(con,new_Vx)
  tot_Vy = np.convolve(con,new_Vy)
  tot_Ax = np.convolve(con,new_Ax)
  tot_Ay = np.convolve(con,new_Ay)

  #Contour
  #--------------------------------------------------------------------------------------
  tot_Sx = np.zeros(1,dtype=float)
  tot_Sy = np.zeros(1,dtype=float)
  tSx=0.
  tSy=0.
  for i in range(int(np.ceil(resolution*tot_Ti[-1]))):
    tSx = tSx + (tot_Vx[i+1]+tot_Vx[i])/2/resolution
    tSy = tSy + (tot_Vy[i+1]+tot_Vy[i])/2/resolution
    tot_Sx = np.hstack([tot_Sx,tSx])
    tot_Sy = np.hstack([tot_Sy,tSy])
  tot_Sx = tot_Sx/1000
  tot_Sy = tot_Sy/1000

  return [tot_Ti,tot_Vx,tot_Vy,tot_Ax,tot_Ay,tot_Sx,tot_Sy]


#model===============================================================
#straight line=======================================================
data_XY = np.array([ [ 0., 0. ],
            [ 150., 0. ],
             ])
radius = np.array([ 0., 0.])

radius = np.reshape(radius,(radius.size,1))
DATA = np.hstack([data_XY, radius])
points = 10000
F=10000
feedrate = np.array([ F for i in range(radius.size) ])
feedrate = np.reshape(feedrate,(feedrate.size,1))
DATA = np.hstack([DATA[:,0:3], feedrate])



[tot_Ti,tot_Ax,tot_Vx,tot_Ay,tot_Vy,xDF,yDF,tot_Timer] = AutoVA( DATA,P1660,P1772,P1783,P1735,P1732,P1737,P1738 )
[tot_Ti,tot_Vx,tot_Vy,tot_Ax,tot_Ay,tot_Sx,tot_Sy] = AfterInterpolation( tot_Ti,tot_Vx,tot_Vy,tot_Ax,tot_Ay,P1769,2 ) #T,Vx,Vy,Ax,Ay,P1769,points/millisecond
Stop_Ti = int(tot_Ti[-1])

matrix_zeros = np.zeros(4000, dtype=int) #建立一個內容為0 長度為10的整數陣列
tot_Ax = np.hstack([tot_Ax,matrix_zeros])
after_tot_Ti = np.linspace(tot_Ti[-1]+tot_Ti[-1]-tot_Ti[-2], tot_Ti[-1]+(tot_Ti[-1]-tot_Ti[-2])*4000, num=4000)
tot_Ti = np.hstack([tot_Ti,after_tot_Ti])

'''#Plot================================================================================
fig =px.line(x = tot_Ti/1000, y = tot_Ax/1600 ,width=1200, height=600)
fig.update_xaxes(title_text='Time(s)')
fig.update_yaxes(title_text='displacement(m)')
fig.update_layout(
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
fig.show()'''

#Force response============================================
##Build transfer function:np_rsdl、np_freq=================
'''fig =px.line(x= tot_Ti,y=tot_Ax,width=600, height=600)
fig.show()'''

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


'''#Plot================================================================================
fig = go.Figure()
#fig =px.line(x=t1,y=np.sum(mode,axis=0),width=600, height=600)
fig.add_trace(go.Scatter(x=modeSum_Y[1], y=modeSum_Y[0]*1000, name='simulation',
                         line=dict(color='firebrick', width=4)))
fig.update_xaxes(title_text='Time(s)')
fig.update_yaxes(title_text='displacement(m)')
fig.update_layout(
    yaxis = dict(
        showexponent = 'all',
        exponentformat = 'e',
        tickfont_family="Arial"
    ),
    font=dict(
        size=18,
        color="black"
        )
)
fig.show()'''

#Read KGM===================================================
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
  
'''fig =px.line(x = df_kgm['Time'],y = df_kgm['A'],width=1200, height=600)
fig.show()  '''

Z_refine = np.zeros(len(df_kgm['Y']))
kgm_stop_idx = np.searchsorted(df_kgm['Time'], Stop_Ti/1000, side="left")


for i in range(kgm_stop_idx):
  Z_refine[i] = i*0.01/kgm_stop_idx
Z_refine[kgm_stop_idx:15000] = 0.01
'''fig =px.line(Z_refine,width=600, height=600)
fig.show() '''



'''Z_refine = np.zeros(len(df_kgm['Y']))
for i in range(5000):
  Z_refine[i] = i*0.0000014+0.003
Z_refine[5000:15000] = 0.01
fig =px.line(Z_refine,width=600, height=600)
fig.show()'''



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




