# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 19:09:24 2023

@author: sl52873
"""

import os
import numpy as np
import nsim2d_recon_p1 as ns   #p1 5phs/   p2 7phs/   p3 10phs

# 20 pixel patterns, spacing ~ 0.36
# 40 pixel patterns, spacing ~ 0.72


fns ='/Users/shaohli/Desktop/Desktop Backup/ReconCode/nsim_simulation_generation/res1.66_15px/sim_nsi2d_ext10ms.tif'
path = os.path.dirname(os.path.abspath(fns))
farme_order = 2                     #1st: satuation pattern are selected. 2nd: Img patterns are selected
p = ns.si2D(fns, farme_order, 6, 5, 5, 0.503, 1.5, 0.0722)
# p = ns.si2D(fns, farme_order, 6, 7, 7, 0.503, 1.5, 0.0722)



"""computation"""

conver=1.57

# x1_1 = np.array([-3.121  + conver, 0.1702])                #10px angle spacing
# x2_1 = np.array([2.641  + conver, 0.1700])
# x3_1 = np.array([2.119  + conver, 0.1700])
# x4_1 = np.array([1.596  + conver, 0.1706])
# x5_1 = np.array([1.071  + conver, 0.1708])
# x6_1 = np.array([0.545  + conver, 0.1707])

# x1_1 = np.array([-3.122 + conver, 0.3408])                # 20px angle spacing
# x2_1 = np.array([2.641 + conver, 0.3392])
# x3_1 = np.array([2.119  + conver, 0.3395])
# x4_1 = np.array([1.597  + conver, 0.3405])
# x5_1 = np.array([1.073  + conver, 0.3419])
# x6_1 = np.array([0.546  + conver, 0.3418])

# x1_1 = np.array([0.020 + conver, 0.2555])                # 15px angle spacing
# x2_1 = np.array([-0.501 + conver, 0.2552])
# x3_1 = np.array([-1.022 + conver, 0.2549])
# x4_1 = np.array([-1.546 + conver, 0.2556])
# x5_1 = np.array([-2.07 + conver, 0.2562])
# x6_1 = np.array([-2.597 + conver, 0.2562])


# x1_1 = np.array([-3.121  + conver, 0.2046])                # 12px angle spacing
# x2_1 = np.array([2.640  + conver, 0.2043])
# x3_1 = np.array([2.118  + conver, 0.2044])
# x4_1 = np.array([1.595  + conver, 0.2049])
# x5_1 = np.array([1.071  + conver, 0.2052])
# x6_1 = np.array([0.544  + conver, 0.2051])

# x1_1 = np.array([-1.57  + conver, 0.1998])                # 12px angle spacing
# x2_1 = np.array([-1.047  + conver, 0.2000])
# x3_1 = np.array([-0.523  + conver, 0.1998])
# x4_1 = np.array([2.404  + conver, 0.1998])
# x5_1 = np.array([0.523  + conver, 0.1998])
# x6_1 = np.array([1.047  + conver, 0.1998])


# x1_1 = np.array([0.022  + conver, 0.6792])                # 40px angle spacing
# x2_1 = np.array([-0.552  + conver, 0.6639])
# x3_1 = np.array([2.100   + conver, 0.6640])
# x4_1 = np.array([1.552   + conver, 0.6801])
# x5_1 = np.array([-2.121  + conver, 0.6645])
# x6_1 = np.array([0.529   + conver, 0.6637])

# x1_1 = np.array([-1.570  + conver, 0.189])                # simulation data
# x2_1 = np.array([-1.047  + conver, 0.189])
# x3_1 = np.array([-0.523  + conver, 0.189])
# x4_1 = np.array([0       + conver, 0.189])
# x5_1 = np.array([0.523   + conver, 0.189])
# x6_1 = np.array([1.048   + conver, 0.189])


x1_1 = np.array([-1.571   + conver, 0.255])                # simulation data
x2_1 = np.array([-1.047   + conver, 0.255])
x3_1 = np.array([-0.524   + conver, 0.255])
x4_1 = np.array([0        + conver, 0.255])
x5_1 = np.array([0.523    + conver, 0.255])
x6_1 = np.array([1.047    + conver, 0.255])



# # # #                                             #p

#1st angle
p.separate(0)
p.shift0()
m1_1 = p.mapoverlap1(x1_1[0], x1_1[1], 1, nps=8, r_ang=0.02, r_sp=0.008)
m1_1 = p.mapoverlap1(m1_1[0], m1_1[1], 1, nps=10, r_ang=0.004, r_sp=0.002)
m1_1 = p.mapoverlap1(m1_1[0], m1_1[1], 1, nps=10, r_ang=0.002, r_sp=0.002)
a1_1 = p.getoverlap1(m1_1[0], m1_1[1], 1)
p.shift1(m1_1[0], m1_1[1])
a1_2 = p.getoverlap2(m1_1[0], m1_1[1]/2, 3)
m1_2 = p.mapoverlap2(m1_1[0], m1_1[1]/2, 3, nps=8, r_ang=0.02, r_sp=0.008)
m1_2 = p.mapoverlap2(m1_2[0], m1_2[1], 3, nps=10, r_ang=0.004, r_sp=0.002)
m1_2 = p.mapoverlap2(m1_2[0], m1_2[1], 3, nps=10, r_ang=0.002, r_sp=0.002)

# 2ed angle
p.separate(1)
p.shift0()
m2_1 = p.mapoverlap1(x2_1[0], x2_1[1], 1, nps=8, r_ang=0.02, r_sp=0.008)
m2_1 = p.mapoverlap1(m2_1[0], m2_1[1], 1, nps=10, r_ang=0.004, r_sp=0.002)
m2_1 = p.mapoverlap1(m2_1[0], m2_1[1], 1, nps=10, r_ang=0.002, r_sp=0.002)
a2_1 = p.getoverlap1(m2_1[0], m2_1[1], 1)
p.shift1(m2_1[0], m2_1[1])
a2_2 = p.getoverlap2(m2_1[0], m2_1[1]/2, 3)
m2_2 = p.mapoverlap2(m2_1[0], m2_1[1]/2, 3, nps=8, r_ang=0.02, r_sp=0.008)
m2_2 = p.mapoverlap2(m2_2[0], m2_2[1], 3, nps=10, r_ang=0.004, r_sp=0.002)
m2_2 = p.mapoverlap2(m2_2[0], m2_2[1], 3, nps=10, r_ang=0.002, r_sp=0.002)


# #3rd angle
p.separate(2)
p.shift0()
m3_1 = p.mapoverlap1(x3_1[0], x3_1[1], 1, nps=8, r_ang=0.02, r_sp=0.008)
m3_1 = p.mapoverlap1(m3_1[0], m3_1[1], 1, nps=10, r_ang=0.004, r_sp=0.002)
m3_1 = p.mapoverlap1(m3_1[0], m3_1[1], 1, nps=10, r_ang=0.002, r_sp=0.002)
a3_1 = p.getoverlap1(m3_1[0], m3_1[1], 1)
p.shift1(m3_1[0], m3_1[1])
a3_2 = p.getoverlap2(m3_1[0], m3_1[1]/2, 3)
m3_2 = p.mapoverlap2(m3_1[0], m3_1[1]/2, 3, nps=8, r_ang=0.02, r_sp=0.008)
m3_2 = p.mapoverlap2(m3_2[0], m3_2[1], 3, nps=10, r_ang=0.004, r_sp=0.002)
m3_2 = p.mapoverlap2(m3_2[0], m3_2[1], 3, nps=10, r_ang=0.002, r_sp=0.002)

#4th angle
p.separate(3)
p.shift0()
m4_1 = p.mapoverlap1(x4_1[0], x4_1[1], 1, nps=8, r_ang=0.02, r_sp=0.008)
m4_1 = p.mapoverlap1(m4_1[0], m4_1[1], 1, nps=10, r_ang=0.004, r_sp=0.002)
m4_1 = p.mapoverlap1(m4_1[0], m4_1[1], 1, nps=10, r_ang=0.002, r_sp=0.002)
a4_1 = p.getoverlap1(m4_1[0], m4_1[1], 1)
p.shift1(m4_1[0], m4_1[1])
a4_2 = p.getoverlap2(m4_1[0], m4_1[1]/2, 3)
m4_2 = p.mapoverlap2(m4_1[0], m4_1[1]/2, 3, nps=8, r_ang=0.02, r_sp=0.008)
m4_2 = p.mapoverlap2(m4_2[0], m4_2[1], 3, nps=10, r_ang=0.004, r_sp=0.002)
m4_2 = p.mapoverlap2(m4_2[0], m4_2[1], 3, nps=10, r_ang=0.002, r_sp=0.002)

#5th angle
p.separate(4)
p.shift0()

m5_1 = p.mapoverlap1(x5_1[0], x5_1[1], 1, nps=8, r_ang=0.02, r_sp=0.008)
m5_1 = p.mapoverlap1(m5_1[0], m5_1[1], 1, nps=10, r_ang=0.004, r_sp=0.002)
m5_1 = p.mapoverlap1(m5_1[0], m5_1[1], 1, nps=10, r_ang=0.002, r_sp=0.002)
a5_1 = p.getoverlap1(m5_1[0], m5_1[1], 1)
p.shift1(m5_1[0], m5_1[1])
a5_2 = p.getoverlap2(m5_1[0], m5_1[1]/2, 3)
m5_2 = p.mapoverlap2(m5_1[0], m5_1[1]/2, 3, nps=8, r_ang=0.02, r_sp=0.008)
m5_2 = p.mapoverlap2(m5_2[0], m5_2[1], 3, nps=10, r_ang=0.004, r_sp=0.002)
m5_2 = p.mapoverlap2(m5_2[0], m5_2[1], 3, nps=10, r_ang=0.002, r_sp=0.002)



#6th angle
#6th angle 1st order
p.separate(5)
p.shift0()
m6_1 = p.mapoverlap1(x6_1[0], x6_1[1], 1, nps=8, r_ang=0.02, r_sp=0.008)
m6_1 = p.mapoverlap1(m6_1[0], m6_1[1], 1, nps=10, r_ang=0.002, r_sp=0.002)
m6_1 = p.mapoverlap1(m6_1[0], m6_1[1], 1, nps=10, r_ang=0.002, r_sp=0.002)
a6_1 = p.getoverlap1(m6_1[0], m6_1[1], 1)
#6th angle 2ed order
p.shift1(m6_1[0], m6_1[1])
a6_2 = p.getoverlap2(m6_1[0], m6_1[1]/2, 3)
m6_2 = p.mapoverlap2(m6_1[0], m6_1[1]/2, 3, nps=8, r_ang=0.02, r_sp=0.008)
m6_2 = p.mapoverlap2(m6_2[0], m6_2[1], 3, nps=10, r_ang=0.002, r_sp=0.002)
m6_2 = p.mapoverlap2(m6_2[0], m6_2[1], 3, nps=10, r_ang=0.002, r_sp=0.002)




# # # #display results
print(m1_1)
print(m1_2)
print(m2_1)
print(m2_2)
print(m3_1)
print(m3_2)
print(m4_1)
print(m4_2)
print(m5_1)
print(m5_2)
print(m6_1)
print(m6_2)

'''save results'''
fn = path + '/' + 'mapoverlap' + str(farme_order) + '.txt'
np.savetxt(fn, ('1st angle', m1_1, m1_2, 
                '2nd angle', m2_1, m2_2,
                '3rd angle', m3_1, m3_2,
                '4th angle', m4_1, m4_2, 
                '5th angle', m5_1, m5_2, 
                '6th angle', m6_1, m6_2, 
                'mapoverlap 1st', m1_1[2], m2_1[2], m3_1[2], m4_1[2], m5_1[2], m6_1[2],
                'mapoverlap 2nd', m1_2[2], m2_2[2], m3_2[2], m4_2[2], m5_2[2], m6_2[2]), fmt='%s')


# '''save results'''
# fn = path + '/' + 'mapoverlap' + str(farme_order) + '.txt'
# np.savetxt(fn, ('1st angle', m1_1, m1_2, 
#                 'mapoverlap 1st', m1_1[2],
#                 'mapoverlap 2nd', m1_2[2]), fmt='%s')










'''reconstruction'''

# m1_1 = np.array([-1.5504, 0.2387])                # angle spacing 12px
# m1_2 = np.array([-1.5504, 0.11935])
# m2_1 = np.array([4.2116, 0.2383])
# m2_2 = np.array([4.2112, 0.11915])
# m3_1 = np.array([3.6895999999999995, 0.23779999999999998])
# m3_2 = np.array([3.6907999999999994, 0.118899999999999991])
# m4_1 = np.array([3.1656, 0.2382])
# m4_2 = np.array([3.1656, 0.1191])
# m5_1 = np.array([2.643, 0.239])
# m5_2 = np.array([2.6433999999999997, 0.1195])
# m6_1 = np.array([2.1166, 0.239])
# m6_2 = np.array([2.117, 0.1195])


# m1_1 = np.array([1.592, 0.2554])                # angle spacing for 'C:/Users/sl52873/Desktop/20230607_10px/cell2/AVG_Substack (2-300-2).tif'
# m1_2 = np.array([1.592, 0.1277])
# m2_1 = np.array([1.0697999999999999, 0.2548])
# m2_2 = np.array([1.0697999999999999, 0.1274])
# m3_1 = np.array([0.5484000000000001, 0.2549])
# m3_2 = np.array([0.5476000000000001, 0.12745])
# m4_1 = np.array([0.025400000000000134, 0.2556])
# m4_2 = np.array([0.025800000000000132, 0.1278])
# m5_1 = np.array([-0.49879999999999974, 0.25589999999999996])
# m5_2 = np.array([-0.49839999999999973, 0.12794999999999998])
# m6_1 = np.array([-1.0259999999999998, 0.2561])
# m6_2 = np.array([-1.0255999999999998, 0.12805])




# verified wiener filter
# p.mu = 1
# p.strength = 0.1
# p.eta = 1
# p.cutoff = 0.15
# # p.eh = 1
# p.eh = []

p.cutoff = 0.0005                   #  10px use 0.0005
p.eta = 0.05
p.mu = 0.20
p.strength = 0.5
# p.eh = 0.5




p.separate(0)
p.shift0()
p.check_components(A=1)
a1_1 = p.getoverlap1(m1_1[0], m1_1[1], 1)
p.shift1(m1_1[0], m1_1[1])
a1_2 = p.getoverlap2(m1_2[0], m1_2[1], 3)

p.separate(1)
p.shift0()
p.check_components(A=2)
a2_1 = p.getoverlap1(m2_1[0], m2_1[1], 1)
p.shift1(m2_1[0], m2_1[1])
a2_2 = p.getoverlap2(m2_2[0], m2_2[1], 3)

p.separate(2)
p.shift0()
p.check_components(A=3)
a3_1 = p.getoverlap1(m3_1[0], m3_1[1], 1)
p.shift1(m3_1[0], m3_1[1])
a3_2 = p.getoverlap2(m3_2[0], m3_2[1], 3)

p.separate(3)
p.shift0()
p.check_components(A=4)
a4_1 = p.getoverlap1(m4_1[0], m4_1[1], 1)
p.shift1(m4_1[0], m4_1[1])
a4_2 = p.getoverlap2(m4_2[0], m4_2[1], 3)

p.separate(4)
p.shift0()
p.check_components(A=5)
a5_1 = p.getoverlap1(m5_1[0], m5_1[1], 1)
p.shift1(m5_1[0], m5_1[1])
a5_2 = p.getoverlap2(m5_2[0], m5_2[1], 3)

p.separate(5)
p.shift0()
p.check_components(A=6)
a6_1 = p.getoverlap1(m6_1[0], m6_1[1], 1)
p.shift1(m6_1[0], m6_1[1])
a6_2 = p.getoverlap2(m6_2[0], m6_2[1], 3)

ang = [m1_1[0], m1_2[0], m2_1[0], m2_2[0], m3_1[0], m3_2[0], m4_1[0], m4_2[0], m5_1[0], m5_2[0], m6_1[0], m6_2[0]]
spacing = [m1_1[1], m1_2[1], m2_1[1], m2_2[1], m3_1[1], m3_2[1], m4_1[1], m4_2[1], m5_1[1], m5_2[1], m6_1[1], m6_2[1]]
phase = [-a1_1[1], -a1_1[1]-a1_2[1], -a2_1[1], -a2_1[1]-a2_2[1], -a3_1[1], -a3_1[1]-a3_2[1], -a4_1[1], -a4_1[1]-a4_2[1], -a5_1[1], -a5_1[1]-a5_2[1], -a6_1[1], -a6_1[1]-a6_2[1]]         #low frequency use this(40px,20px,12px)     
mag = 1* [1, 1.5, 
          1, 1.5, 
          1, 1.5, 
          1, 1.5, 
          1, 1.5, 
          1, 1.5]

# ang = [m1_1[0], m1_2[0]]
# spacing = [m1_1[1], m1_2[1]]
# phase = [-a1_1[1], -a1_1[1]-a1_2[1]]         #low frequency use this(40px,20px,12px)     
# mag = 1* [1, 1.5]


path1 = 'Nsim_1st_frame'
path2 = 'Nsim_2nd_frame'
pathsim1 = '2d_sim_1'
pathsim2 = '2d_sim_2'

if farme_order == 1 :
    if os.path.exists(path + '/' + path1) == False:
        os.mkdir(path + '/' + path1)
    if os.path.exists(path + '/' + pathsim1) == False:
        os.mkdir(path + '/' + pathsim1)
elif farme_order == 2 : 
    if os.path.exists(path + '/' + path2) == False:
        os.mkdir(path + '/' + path2)
    if os.path.exists(path + '/' + pathsim2) == False:
        os.mkdir(path + '/' + pathsim2)

p.recon(6, ang, spacing, phase, mag, path1, path2, zero_order=True)
p.recon_sim(6, ang, spacing, phase, mag, pathsim1, pathsim2, zero_order=True)

