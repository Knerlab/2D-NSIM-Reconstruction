# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 13:23:33 2024

@author: sl52873
"""


import os
import numpy as np
import nsim2d_recon_p2 as ns   #p1 5phs/      p2 7phs/     p3 10phs
# import nsim2d_recon_p3 as ns   #p1 5phs/     p2 7phs/    p3 10phs

# 20 pixel patterns, spacing ~ 0.36
# 40 pixel patterns, spacing ~ 0.72


fns = 'C:/Users/sl52873/Desktop/Recon_result/rsEGFP2/20240919_20px/cell10/AVG_Substack (2-960-2).tif'
path = os.path.dirname(os.path.abspath(fns))
farme_order = 2                     #1st: satuation pattern are selected. 2nd: Img patterns are selected

p = ns.si2D(fns, farme_order, 6, 10, 7, 0.503, 1.5, 0.0722)
# p = ns.si2D(fns, farme_order, 5, 5, 5, 0.488, 1.2, 0.089)




"""computation"""

conver=1.57

# x1_1 = np.array([3.128  + conver, 0.1710])                #10px angle spacing
# x2_1 = np.array([2.606  + conver, 0.1706])
# x3_1 = np.array([2.083  + conver, 0.1706])
# x4_1 = np.array([1.561  + conver, 0.1708])
# x5_1 = np.array([1.037  + conver, 0.1713])
# x6_1 = np.array([0.512  + conver, 0.1712])

x1_1 = np.array([-3.122 + conver, 0.3408])                # 20px angle spacing
x2_1 = np.array([2.641 + conver, 0.3392])
x3_1 = np.array([2.119  + conver, 0.3395])
x4_1 = np.array([1.597  + conver, 0.3405])
x5_1 = np.array([1.073  + conver, 0.3419])
x6_1 = np.array([0.546  + conver, 0.3418])

# x1_1 = np.array([-3.122 + conver, 0.2555])                # 15px angle spacing
# x2_1 = np.array([2.641 + conver, 0.2551])
# x3_1 = np.array([2.119 + conver, 0.2552])
# x4_1 = np.array([1.596 + conver, 0.2558])
# x5_1 = np.array([1.072 + conver, 0.2562])
# x6_1 = np.array([0.544 + conver, 0.2560])

# x1_1 = np.array([-3.121 + conver, 0.2383])                # 14px angle spacing
# x2_1 = np.array([2.640 + conver, 0.2383])
# x3_1 = np.array([2.118 + conver, 0.2382])
# x4_1 = np.array([1.594 + conver, 0.2387])
# x5_1 = np.array([1.071 + conver, 0.2394])
# x6_1 = np.array([0.545 + conver, 0.2390])


# x1_1 = np.array([3.124  + conver, 0.2051])                # 12px angle spacing
# x2_1 = np.array([2.602  + conver, 0.2045])
# x3_1 = np.array([2.080  + conver, 0.2047])
# x4_1 = np.array([1.557  + conver, 0.2052])
# x5_1 = np.array([1.033  + conver, 0.2056])
# x6_1 = np.array([0.506  + conver, 0.2055])

# x1_1 = np.array([-1.57  + conver, 0.1998])                # 12px angle spacing
# x2_1 = np.array([-1.047  + conver, 0.2000])
# x3_1 = np.array([-0.523  + conver, 0.1998])
# x4_1 = np.array([2.404  + conver, 0.1998])
# x5_1 = np.array([0.523  + conver, 0.1998])
# x6_1 = np.array([1.047  + conver, 0.1998])


# x1_1 = np.array([-0.012  + conver, 0.6796])                # 40px angle spacing
# x2_1 = np.array([-0.552  + conver, 0.6639])
# x3_1 = np.array([2.100   + conver, 0.6640])
# x4_1 = np.array([1.552   + conver, 0.6801])
# x5_1 = np.array([-2.121  + conver, 0.6645])
# x6_1 = np.array([0.529   + conver, 0.6637])

# x1_1 = np.array([-1.570  + conver, 0.255])                # simulation data
# x2_1 = np.array([-1.047  + conver, 0.255])
# x3_1 = np.array([-0.523  + conver, 0.255])
# x4_1 = np.array([0       + conver, 0.255])
# x5_1 = np.array([0.523   + conver, 0.255])
# x6_1 = np.array([1.048   + conver, 0.255])


# x1_1 = np.array([-1.571  + conver, 0.238])                # simulation data
# x2_1 = np.array([-0.314  + conver, 0.238])
# x3_1 = np.array([0.942   + conver, 0.238])
# x4_1 = np.array([2.199   + conver, 0.238])
# x5_1 = np.array([-2.827  + conver, 0.238])





p.cutoff = 0.0002                    #  10px use 0.0005
p.eta = 0.06
p.mu = 0.24

# # #                                             #p

# 1st angle
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
p.shift2(m1_2[0], m1_2[1])
a1_3 = p.getoverlap3(m1_2[0], m1_1[1]/3, 5)
m1_3 = p.mapoverlap3(m1_2[0], m1_1[1]/3, 5, nps=8, r_ang=0.02, r_sp=0.008)
m1_3 = p.mapoverlap3(m1_3[0], m1_3[1], 5, nps=10, r_ang=0.004, r_sp=0.002)
m1_3 = p.mapoverlap3(m1_3[0], m1_3[1], 5, nps=10, r_ang=0.002, r_sp=0.002)

#2ed angle
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
p.shift2(m2_2[0], m2_2[1])
a2_3 = p.getoverlap3(m2_2[0], m2_1[1]/3, 5)
m2_3 = p.mapoverlap3(m2_2[0], m2_1[1]/3, 5, nps=8, r_ang=0.02, r_sp=0.008)
m2_3 = p.mapoverlap3(m2_3[0], m2_3[1], 5, nps=10, r_ang=0.004, r_sp=0.002)
m2_3 = p.mapoverlap3(m2_3[0], m2_3[1], 5, nps=10, r_ang=0.002, r_sp=0.002)


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
p.shift2(m3_2[0], m3_2[1])
a3_3 = p.getoverlap3(m3_2[0], m3_1[1]/3, 5)
m3_3 = p.mapoverlap3(m3_2[0], m3_1[1]/3, 5, nps=8, r_ang=0.02, r_sp=0.008)
m3_3 = p.mapoverlap3(m3_3[0], m3_3[1], 5, nps=10, r_ang=0.004, r_sp=0.002)
m3_3 = p.mapoverlap3(m3_3[0], m3_3[1], 5, nps=10, r_ang=0.002, r_sp=0.002)

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
p.shift2(m4_2[0], m4_2[1])
a4_3 = p.getoverlap3(m4_2[0], m4_1[1]/3, 5)
m4_3 = p.mapoverlap3(m4_2[0], m4_1[1]/3, 5, nps=8, r_ang=0.02, r_sp=0.008)
m4_3 = p.mapoverlap3(m4_3[0], m4_3[1], 5, nps=10, r_ang=0.004, r_sp=0.002)
m4_3 = p.mapoverlap3(m4_3[0], m4_3[1], 5, nps=10, r_ang=0.002, r_sp=0.002)

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
p.shift2(m5_2[0], m5_2[1])
a5_3 = p.getoverlap3(m5_2[0], m5_1[1]/3, 5)
m5_3 = p.mapoverlap3(m5_2[0], m5_1[1]/3, 5, nps=8, r_ang=0.02, r_sp=0.008)
m5_3 = p.mapoverlap3(m5_3[0], m5_3[1], 5, nps=10, r_ang=0.004, r_sp=0.002)
m5_3 = p.mapoverlap3(m5_3[0], m5_3[1], 5, nps=10, r_ang=0.002, r_sp=0.002)




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
p.shift2(m6_2[0], m6_2[1])
a6_3 = p.getoverlap3(m6_2[0], m6_1[1]/3, 5)
m6_3 = p.mapoverlap3(m6_2[0], m6_1[1]/3, 5, nps=8, r_ang=0.02, r_sp=0.008)
m6_3 = p.mapoverlap3(m6_3[0], m6_3[1], 5, nps=10, r_ang=0.004, r_sp=0.002)
m6_3 = p.mapoverlap3(m6_3[0], m6_3[1], 5, nps=10, r_ang=0.002, r_sp=0.002)





# #display results
print(m1_1)
print(m1_2)
print(m1_3)
print(m2_1)
print(m2_2)
print(m2_3)
print(m3_1)
print(m3_2)
print(m3_3)
print(m4_1)
print(m4_2)
print(m4_3)
print(m5_1)
print(m5_2)
print(m5_3)
print(m6_1)
print(m6_2)
print(m6_3)

'''save results'''
fn = path + '/' + 'mapoverlap' + str(farme_order) + '.txt'
np.savetxt(fn, ('1st angle', m1_1, m1_2, m1_3,
                '2nd angle', m2_1, m2_2, m2_3,
                '3rd angle', m3_1, m3_2, m3_3,
                '4th angle', m4_1, m4_2, m4_3,
                '5th angle', m5_1, m5_2, m5_3,
                '6th angle', m6_1, m6_2, m6_3,
                'mapoverlap 1st', m1_1[2], m2_1[2], m3_1[2], m4_1[2], m5_1[2], m6_1[2],
                'mapoverlap 2nd', m1_2[2], m2_2[2], m3_2[2], m4_2[2], m5_2[2], m6_2[2],
                'mapoverlap 3nd', m1_3[2], m2_3[2], m3_3[2], m4_3[2], m5_3[2], m6_3[2]), fmt='%s')









'''reconstruction'''

# m1_1 = np.array([0.0, 0.1798])                # angle spacing 12px
# m1_2 = np.array([0.0, 0.0899])
# m2_1 = np.array([1.0466000000000002, 0.1798])
# m2_2 = np.array([1.0470000000000002, 0.0899])
# m3_1 = np.array([2.0942, 0.1802])
# m3_2 = np.array([2.0942, 0.0901])
# m4_1 = np.array([3.142, 0.1798])
# m4_2 = np.array([3.142, 0.0899])
# m5_1 = np.array([4.189, 0.1798])
# m5_2 = np.array([4.189, 0.0899])
# m6_1 = np.array([-1.0473999999999999, 0.1802])
# m6_2 = np.array([-1.0473999999999999, 0.0901])



# verified wiener filter
p.mu = 0.24
p.fwhm = 0.99
p.strength = 0.1
p.minv = 0.
p.eta = 0.06
p.cutoff = 0.0002




p.separate(0)
p.shift0()
p.check_components(A=1)
a1_1 = p.getoverlap1(m1_1[0], m1_1[1], 1)
p.shift1(m1_1[0], m1_1[1])
a1_2 = p.getoverlap2(m1_2[0], m1_2[1], 3)
p.shift2(m1_2[0], m1_2[1])
a1_3 = p.getoverlap3(m1_3[0], m1_3[1], 5)

p.separate(1)
p.shift0()
p.check_components(A=2)
a2_1 = p.getoverlap1(m2_1[0], m2_1[1], 1)
p.shift1(m2_1[0], m2_1[1])
a2_2 = p.getoverlap2(m2_2[0], m2_2[1], 3)
p.shift2(m2_2[0], m2_2[1])
a2_3 = p.getoverlap3(m2_3[0], m2_3[1], 5)

p.separate(2)
p.shift0()
p.check_components(A=3)
a3_1 = p.getoverlap1(m3_1[0], m3_1[1], 1)
p.shift1(m3_1[0], m3_1[1])
a3_2 = p.getoverlap2(m3_2[0], m3_2[1], 3)
p.shift2(m3_2[0], m3_2[1])
a3_3 = p.getoverlap3(m3_3[0], m3_3[1], 5)

p.separate(3)
p.shift0()
p.check_components(A=4)
a4_1 = p.getoverlap1(m4_1[0], m4_1[1], 1)
p.shift1(m4_1[0], m4_1[1])
a4_2 = p.getoverlap2(m4_2[0], m4_2[1], 3)
p.shift2(m4_2[0], m4_2[1])
a4_3 = p.getoverlap3(m4_3[0], m4_3[1], 5)

p.separate(4)
p.shift0()
p.check_components(A=5)
a5_1 = p.getoverlap1(m5_1[0], m5_1[1], 1)
p.shift1(m5_1[0], m5_1[1])
a5_2 = p.getoverlap2(m5_2[0], m5_2[1], 3)
p.shift2(m5_2[0], m5_2[1])
a5_3 = p.getoverlap3(m5_3[0], m5_3[1], 5)

p.separate(5)
p.shift0()
p.check_components(A=6)
a6_1 = p.getoverlap1(m6_1[0], m6_1[1], 1)
p.shift1(m6_1[0], m6_1[1])
a6_2 = p.getoverlap2(m6_2[0], m6_2[1], 3)
p.shift2(m6_2[0], m6_2[1])
a6_3 = p.getoverlap3(m6_3[0], m6_3[1], 5)

ang = [m1_1[0], m1_2[0], m1_3[0], m2_1[0], m2_2[0], m2_3[0], m3_1[0], m3_2[0], m3_3[0],m4_1[0], m4_2[0], m4_3[0], m5_1[0], m5_2[0], m5_3[0], m6_1[0], m6_2[0], m6_3[0]]
spacing = [m1_1[1], m1_2[1], m1_3[1], m2_1[1], m2_2[1], m2_3[1], m3_1[1], m3_2[1], m3_3[1], m4_1[1], m4_2[1], m4_3[1], m5_1[1], m5_2[1], m5_3[1], m6_1[1], m6_2[1], m6_3[1]]
phase = [-a1_1[1], -a1_1[1]-a1_2[1], -a1_1[1]-a1_2[1]-a1_3[1], -a2_1[1], -a2_1[1]-a2_2[1], -a2_1[1]-a2_2[1]-a2_3[1], -a3_1[1], -a3_1[1]-a3_2[1], -a3_1[1]-a3_2[1]-a3_3[1], -a4_1[1], -a4_1[1]-a4_2[1], -a4_1[1]-a4_2[1]-a4_3[1], -a5_1[1], -a5_1[1]-a5_2[1], -a5_1[1]-a5_2[1]-a5_3[1], -a6_1[1], -a6_1[1]-a6_2[1], -a6_1[1]-a6_2[1]-a6_3[1]]         #low frequency use this(40px,20px,12px)     
mag = 1* [1, 3, 5, 
          1, 3, 5, 
          1, 3, 5, 
          1, 3, 5, 
          1, 3, 5, 
          1, 3, 5]


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
        
p.recon_sim(6, ang, spacing, phase, mag, pathsim1, pathsim2, zero_order=True)
p.recon_1order(6, ang, spacing, phase, mag, path1, path2, zero_order=True)
p.recon_2order(6, ang, spacing, phase, mag, path1, path2, zero_order=True)


