"""
Author: Amogh Jalihal
Date: 2025-04-08
"""
import sys
sys.path.append('./')
import model.dual_siderophore_independent_breakdown as model
import os

import matplotlib
from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
matplotlib.rcParams.update({'font.size': 18})


### Hard coded parameters. See supplement
p_K_m1,p_K_m2 = [1e-06, 1e-02]
p_K = p_K_m1
p_Y_glc = 1.2
p_Y_fe = 1.2e-5
p_R= 30.e-6
p_mu_max = 1.4
p_G0 = 0.9 
p_Y_sid= 10   ## Arbitrary conversion
gridsize = 200

######################################################################
## Model evaluation
print(f"Evaluating steady state at ({gridsize}x{gridsize}) inputs. This can take a few seconds...")
pD, pM = np.meshgrid(np.linspace(0.01,1.3,gridsize), np.linspace(0.01,1,gridsize))
fe, glc, siderophore, cell = model.steadystate(pD, pM,p_mu_max,p_K_m1,p_K_m2,
                                                    p_K,p_Y_fe,p_Y_glc,p_Y_sid,p_R,p_G0)
partialmixed = model.mixedpartial(pD, pM,p_mu_max,
                                   p_K_m1,p_K_m2,
                                   p_K,p_Y_fe,p_Y_glc,p_Y_sid,p_R,p_G0)
######################################################################

######################################################################
## Numerically approximate the siderophore optimality and cell washout curves
Dcollect, Mcollect = [], []
for drow, mrow, crow in zip(pD, pM, cell):
    for d, m, c in zip(drow,mrow,crow):
        if c <= 0:
            Dcollect.append(d)
            Mcollect.append(m)
            break
Dmax, Mmax = [], []
for drow, mrow, prow in zip(pD, pM, partialmixed):
    for d, m, p in zip(drow,mrow,prow):
        if (p == max(prow)) and max(prow) > 0:
            if d < p_mu_max:
                Dmax.append(d)
                Mmax.append(m)
            break
######################################################################

######################################################################
#### FIGURES



print("PLOT: 1. Siderophore concentration")
fig = plt.figure(figsize=(8,8))
ax =fig.add_subplot(1,1,1)
ax.plot(Dmax, Mmax, 'w--',lw=2,alpha=0.5)

p = ax.pcolormesh(pD, pM, siderophore)
ax.plot(Dcollect, Mcollect, 'r--')
fig.colorbar(p, ax=ax)
ax.text(0.8,0.35,"$(\\partial^2 P/\\partial D\\partial M)_{\\text{max}} $",
        color="w",
        rotation=50)
ax.text(0.9,0.1,"Cell washout",
        color="r",
        rotation=45)
ax.set_xlabel("D (hr-1)")
ax.set_ylabel("M (g)")
ax.set_title("Optimal siderophore production")
plt.tight_layout()
plt.savefig("./fig/fig1efg_optimal_siderophore.png",dpi=200,bbox_inches="tight")
plt.savefig("./fig/fig1efg_optimal_siderophore.pdf",dpi=200,bbox_inches="tight")
plt.close("all")


print("PLOT: 2. Free glucose concentrations")
fig = plt.figure(figsize=(8,8))
ax =fig.add_subplot(1,1,1)
#ax.plot(Dmax, Mmax, 'w--',lw=2,alpha=0.5)
p = ax.pcolormesh(pD, pM, np.log10(glc), )
ax.plot(Dcollect, Mcollect, 'r--')
fig.colorbar(p, ax=ax)
# ax.text(0.8,0.35,"$(\\partial^2 P/\\partial D\\partial M)_{\\text{max}} $",
#         color="w",
#         rotation=50)
ax.text(0.9,0.1,"Cell washout",
        color="r",
        rotation=45)
ax.set_xlabel("D (hr-1)")
ax.set_ylabel("M (g)")
ax.set_title("Log10(Glucose concentration)")
plt.tight_layout()
plt.savefig("./fig/fig1efg_free_glucose.pdf",dpi=200,bbox_inches="tight")
plt.savefig("./fig/fig1efg_free_glucose.png",dpi=200,bbox_inches="tight")
plt.close("all")

print("PLOT: 3. Cell density")
fig = plt.figure(figsize=(8,8))
ax =fig.add_subplot(1,1,1)
#ax.plot(Dmax, Mmax, 'w--',lw=2,alpha=0.5)
p = ax.pcolormesh(pD, pM, cell, vmin=0.001)
ax.plot(Dcollect, Mcollect, 'r--')
fig.colorbar(p, ax=ax)
# ax.text(0.8,0.35,"$(\\partial^2 P/\\partial D\\partial M)_{\\text{max}} $",
#         color="w",
#         rotation=50)
ax.text(0.9,0.1,"Cell washout",
        color="r",
        rotation=45)
ax.set_xlabel("D (hr-1)")
ax.set_ylabel("M (g)")
ax.set_title("Cell density")
plt.tight_layout()
plt.savefig("./fig/fig1efg_cell.pdf",dpi=200,bbox_inches="tight")
plt.savefig("./fig/fig1efg_cell.png",dpi=200,bbox_inches="tight")
plt.close("all")

print("PLOT: 4. Free iron concentrations")
fig = plt.figure(figsize=(8,8))
ax =fig.add_subplot(1,1,1)
p = ax.pcolormesh(pD, pM, np.log10(fe))
ax.plot(Dcollect, Mcollect, 'r--')
fig.colorbar(p, ax=ax)
ax.text(0.9,0.1,"Cell washout",
        color="r",
        rotation=45)
ax.set_xlabel("D (hr-1)")
ax.set_ylabel("M (g)")
ax.set_title("Log10(Iron concentration)")
plt.tight_layout()
plt.savefig("./fig/fig1efg_free_iron.pdf",dpi=200,bbox_inches="tight")
plt.savefig("./fig/fig1efg_free_iron.png",dpi=200,bbox_inches="tight")
plt.close("all")


print("PLOT: 5. Dual limitation regimes")
Fe = fe/(p_K_m1 + fe)
Glc = glc/(p_K_m2 + glc)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
p = ax.pcolormesh(pD, pM, np.log10(Fe/Glc), vmin=-2,vmax=2,cmap="bwr_r")
fig.colorbar(p,ax=ax)
ax.plot(Dcollect, Mcollect, 'r--')
ax.set_xlabel("D")
ax.set_ylabel("M")
ax.set_title("(fe/(p_K_m1 + fe))/(glc/(p_K_m2 + glc))")
plt.tight_layout()
plt.savefig("./fig/dual-limitation.png")
plt.close("all")
