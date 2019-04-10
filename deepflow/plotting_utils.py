import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt 
import xarray as xr
import pandas as pd
import numpy as np
import scipy.io as io
import os
import scipy.stats
from skimage import measure
from skimage.morphology import binary_dilation
from scipy import stats
from tqdm import tqdm as tqdm 
import matplotlib 
import matplotlib.colors as colors
import matplotlib.cm as cmx

def update_matplotlib_config():
    # inspired by http://nipunbatra.github.io/2014/08/latexify/
    params = {
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'image.origin': 'lower',
        'image.interpolation': 'nearest',
        'image.cmap': 'gray',
        'axes.grid': False,
        'savefig.dpi': 150,  # to adjust notebook inline plot size
        'axes.labelsize': 8, # fontsize for x and y labels (was 10)
        'axes.titlesize': 8,
        'font.size': 8, # was 10
        'legend.fontsize': 10, # was 10
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'text.usetex': True,
        'figure.figsize': [14, 12],
        'font.family': 'serif',
    }
    matplotlib.rcParams.update(params)
    return True

def load_gt_model_from_numpy(file):
    gt = np.load(file)
    return gt[0, :, ::-1].T

def get_reference_curves(working_dir, perm, case='/reference/reference_noise.nc'):
    ds_ref = xr.open_dataset(working_dir+perm+case)
    qor_ref = ds_ref['state_variables'][dict(state_variable=2, well=1)]*(-60*60*24)
    qwr_ref = ds_ref['state_variables'][dict(state_variable=1, well=1)]*(-60*60*24)
    p_ref = ds_ref['state_variables'][dict(state_variable=0, well=0)]/1e5

    return [qor_ref, qwr_ref, p_ref]

def extract_curves(folder, N=500):
    qor_curves = []
    qwr_curves = []
    p_curves = []
    perms, poros = [], []
    material_grads = []
    for i in tqdm(range(0, N)):
        try:
            ds = xr.open_dataset(folder+'/iteration_'+str(i)+'.nc')
            qor_curves.append(ds['state_variables'][dict(state_variable=2, well=1)]*(-60*60*24))
            qwr_curves.append(ds['state_variables'][dict(state_variable=1, well=1)]*(-60*60*24))
            p_curves.append(ds['state_variables'][dict(state_variable=0, well=0)]/1e5)
            perms.append(ds['material_properties'][1].values)
            poros.append(ds['material_properties'][0].values)
            material_grads.append(ds['material_derivatives'].values)
            ds.close()
        except FileNotFoundError or KeyError:
            pass
    return np.array(qor_curves), np.array(qwr_curves), np.array(p_curves), np.array(perms), np.array(material_grads), np.array(poros)

def inverse_permeability_transform(x, a=0.001, b=1e-12):
    return x/b-a

def plot_colored_rate_curves(axarr, min_curves, iterations_to_show, ref_curves, dts, method="-Adam", color="blue", alpha=0.1, ref_color="red", cmap="coolwarm"):  
        
    selected_curves = min_curves[iterations_to_show]

    jet = cm = plt.get_cmap(cmap) 
    log_total = iterations_to_show
    cNorm  = colors.Normalize(vmin=np.min(log_total), vmax=np.max(log_total))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    cs = [scalarMap.to_rgba(j) for j in log_total]

    for j, (curves, idx) in enumerate(zip(selected_curves, iterations_to_show)):

        if j == len(selected_curves)-1:
            for k in range(3):
                axarr[k].plot(dts, curves[k], color=cs[j], alpha=0.5, label="Simulated") 
        else:
            for k in range(3):
                cbar_handle = axarr[k].plot(dts, curves[k], color=cs[j], alpha=0.5)   
    

    for k in range(3):
        axarr[k].plot(dts, selected_curves[0][k], color='black', linewidth=2, linestyle="-.", alpha=1.0, label="Initial")     

    for k in range(3):
        axarr[k].plot(dts, min_curves[iterations_to_show[-1]][k], color='black', alpha=1.0, linewidth=4, linestyle="--", label=r"Minimum $\mathcal{L}(\mathbf{z})$")  
                
    for i, t, loc  in zip(range(3), ["Oil Rate [m3/d]", "Water Rate [m3/d]", "Pressure [Bar]"], [0, 1, 1]):
        axarr[i].scatter(dts, ref_curves[i], color=ref_color, marker="x", s=100, label="Observed", zorder=100)
        axarr[i].set_ylabel(t, fontsize=16)
        axarr[i].legend(fontsize=20, loc=loc)

    for a in axarr:
        a.set_xlabel("Time [days]", fontsize=16)
    
    axarr[0].set_ylim(-5, 325)
    axarr[1].set_ylim(-5, 325)
    axarr[2].set_yscale("log")
    axarr[2].set_ylim(150, 100000)

    scalarMap._A = []
    plt.colorbar(scalarMap)
    for a, label, left, up in zip(axarr.flatten(), ["a)", "b)", "c)"], [-30]*3, [330, 330, 1.15e5]*3):
        a.text(left, up, label, fontsize=22)    

    axarr[2].text(800, 2e4, r"Optimization Iteration", rotation=90, fontsize=24)

    return True
    


def plot_rate_curves(axarr, min_curves, ref_curves, dts, method="-Adam", color="blue", alpha=0.05, ref_color="red"):
    for j, curves in enumerate(min_curves):
        if j == len(min_curves)-1:
            for k in range(3):
                axarr[k].plot(dts, curves[k], color=color, alpha=alpha, label="Simulated")       
        else:
            for k in range(3):
                axarr[k].plot(dts, curves[k], color=color, alpha=alpha)   

    for i in range(3):
        axarr[i].plot(dts, np.mean(min_curves[:, i], axis=0), linewidth=2, linestyle="-.", color="green", label="Mean")

    for a in axarr:
        a.set_xlabel("Time [days]")
    
    axarr[0].set_ylim(-5, 325)
    axarr[1].set_ylim(-5, 325)
    axarr[2].set_yscale("log")
    axarr[2].set_ylim(150, 20000)

    for i, t, s  in zip(range(3), ["Oil Rate [m3/d]", "Water Rate [m3/d]", "Pressure [Bar]"], [8, 8, 20]):
        axarr[i].scatter(dts, ref_curves[i], color=ref_color, label="Observed", marker="x", s=s, alpha=1.0)
        axarr[i].set_ylabel(t)
        axarr[i].legend()


def plot_rate_bounds(axarr, min_curves, ref_curves, dts, method="-Adam"):
    mean_curves, lower, upper = mean_confidence_interval(min_curves)
    
    for i, t, s  in zip(range(3), ["Oil Rate [m3/d]", "Water Rate [m3/d]", "Pressure [Bar]"], [8, 8, 20]):
        axarr[i].plot(dts, mean_curves[i], color="black", linestyle="-", label="Avg. Simulated")
        axarr[i].plot(dts, upper[i], color="black", linestyle="--")
        axarr[i].plot(dts, lower[i], color="black", linestyle="--")
        axarr[i].fill_between(dts, lower[i], upper[i], color="gray", alpha=0.5, label=r"Mean $\pm 95\%$ conf.")

        axarr[i].scatter(dts, ref_curves[i], color="red", marker="x", label="Observed", s=s)
        axarr[i].set_ylabel(t)
        axarr[i].legend()

    for a in axarr:
        a.set_xlabel("Time [days]")
        
    axarr[0].set_ylim(-5, 325)
    axarr[1].set_ylim(-5, 325)
    axarr[2].set_yscale("log")
    axarr[2].set_ylim(150, 20000)
        
def plot_facies(axarr, min_poroperms, envelope=None):
    x = np.where(min_poroperms[:, 1]>1e-13, 1, 0)
    mean = x.mean(axis=0)[::-1]
    std = x.std(axis=0)[::-1]
    sx1 = axarr[0].imshow(mean, vmin=0, vmax=1)
    axarr[0].set_xticks([])
    axarr[0].set_yticks([])
    axarr[0].set_ylabel("Mean", fontsize=12)
    colorbar(sx1)
    
    sx2 = axarr[1].imshow(std, vmin=0, vmax=0.5)
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])
    axarr[1].set_ylabel("Std. Dev.", fontsize=12)
    colorbar(sx2)
    
    if envelope is not None:
        axarr[0].contour(envelope, colors="r", linewidths=(0.2, ), alpha=0.5)
        axarr[1].contour(envelope, colors="r", linewidths=(0.2, ), alpha=0.5)

def plot_row_envelopes(i, properties, curves, ref_curves, dts, envelope=None, desc=""):
    ax1 = plt.subplot2grid((8, 4*2), (i, 0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid((8, 4*2), (i+1, 0), rowspan=1, colspan=2)
    ax2.annotate(desc, xy=(0, 0.0), xytext=(-0.25, 1.5), textcoords='axes fraction', 
                 rotation=90, fontsize=14)
    plot_facies([ax1, ax2], properties, envelope)
    
    ax3 = plt.subplot2grid((8, 4*2), (i, 2), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid((8, 4*2), (i, 4), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid((8, 4*2), (i, 6), rowspan=2, colspan=2)
    plot_rate_bounds([ax3, ax4, ax5], curves, ref_curves, dts)
    
def plot_row_curves(i, properties, curves, ref_curves, dts, envelope=None, desc=""):
    ax1 = plt.subplot2grid((8, 4*2), (i, 0), rowspan=1, colspan=2)
    ax2 = plt.subplot2grid((8, 4*2), (i+1, 0), rowspan=1, colspan=2)
    ax2.annotate(desc, xy=(0, 0.0), xytext=(-0.25, 1.5), textcoords='axes fraction', 
                 rotation=90, fontsize=14)
    plot_facies([ax1, ax2], properties, envelope)
    
    ax3 = plt.subplot2grid((8, 4*2), (i, 2), rowspan=2, colspan=2)
    ax4 = plt.subplot2grid((8, 4*2), (i, 4), rowspan=2, colspan=2)
    ax5 = plt.subplot2grid((8, 4*2), (i, 6), rowspan=2, colspan=2)
    plot_rate_curves([ax3, ax4, ax5], curves, ref_curves, dts)

def to_deltas(dt):
    dts = [dt[0]]
    for i in range(1, len(dt)):
        dts.append(dts[i-1]+dt[i])
    return np.array(dts)

def create_simulation_time_axis():
    dt_1 = to_deltas([1, 1, 3, 5, 5, 10, 10, 10, 15, 15, 15, 15, 15, 15, 15])
    dt_2 = 150+to_deltas(np.array([15]*10))
    dt_3 = 300+to_deltas([25]*6)
    dt_4 = 450+to_deltas([25]*6)
    dts = np.concatenate([dt_1, dt_2, dt_3, dt_4])
    return dts

def extract_min_misfits(misfits, pos):
    mins = np.array([(i, np.argmin(x[:, pos], axis=0), x[np.argmin(x[:, pos], axis=0), pos]) for i, x in enumerate(misfits) if len(x) != 0])
    return mins

def load_folders(working_dir, folders, functional="min_f"):
    temp = []
    temp_poroperms = []
    temp_zs = []
    for folder in folders:
        min_f_curves = np.load(os.path.join(working_dir, folder, functional+"_curves.npy"))
        min_f_poroperms = np.load(os.path.join(working_dir, folder, functional+"_poroperms.npy"))
        min_f_zs = np.load(os.path.join(working_dir, folder, functional+"_zs.npy"))
        temp.append([min_f_curves])
        temp_poroperms.append([min_f_poroperms])
        temp_zs.append([min_f_zs])
    return temp, temp_poroperms, temp_zs 

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def get_unconditionals(working_dir, perm, N=1000):
    curves = []
    poros, perms = [], []
    zs = []
    misfits = []
    for i in range(0, N):
        try:
            folder = os.path.join(working_dir, perm, 'unconditional_noise/run_'+str(i))
            ds = xr.open_dataset(folder+'/iteration_0.nc')
            qo = ds['state_variables'][dict(state_variable=2, well=1)]*(-60*60*24)
            qw = ds['state_variables'][dict(state_variable=1, well=1)]*(-60*60*24)
            p = ds['state_variables'][dict(state_variable=0, well=0)]/1e5
            poros.append(ds['material_properties'][0].values)
            perms.append(ds['material_properties'][1].values)
            curves.append([qo, qw, p])
            zs.append([ds['latent_variables'].values])
            misfits.append([ds['misfit_value'].values])
            ds.close()
        except FileNotFoundError:
            print(i, " not found ")
        if i % 100 == 99:
            print(i)
    return np.array(curves), np.array(poros), np.array(perms), np.array(zs), np.array(misfits)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = data.shape[0]
    m, se = np.mean(a, axis=0), scipy.stats.sem(a, axis=0)
    h = 2*np.std(a, axis=0)#se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def determine_connected(facies, dilation=False):
    if dilation:
        facies = binary_dilation(facies)
    connected = measure.label(facies[::-1], background=0)
    labels = np.unique(connected)
    for label in labels[1:]:
        cluster_only = np.where(connected == label, 1, 0)
        well_a = cluster_only[:, 8:10]
        well_b = cluster_only[:, 119:121]
        if np.sum(well_a) > 0 and np.sum(well_b) > 0:
            return True
    return False

def plot_misfit_histograms(axarr, misfits):
    qo_error = pd.DataFrame([m[:, 0] for m in misfits])
    qw_error = pd.DataFrame([m[:, 1] for m in misfits])
    p_error = pd.DataFrame([m[:, 2] for m in misfits])
    f_error = pd.DataFrame([m[:, 3] for m in misfits])
    well_acc = pd.DataFrame([m[:, 4] for m in misfits])
    t_error = pd.DataFrame([m[:, -1] for m in misfits])

    threshs = []
    error_threshs = [1e4, 1e3, 1e2, 1e1, 1e0]
    for t in error_threshs:
        temp = []
        for row in f_error.values:
            minim = [np.NaN, np.NaN]
            for i, value in enumerate(row):
                if value <= t:
                    minim = [i, value]
                    break
            temp.append(minim) 
        threshs.append(temp)
    threshs = np.array(threshs)

    for idx_thresh, ax, e, m in zip(range(1, 4), axarr, [[1, 3], [1, 2], [1, 1]], [100, 100, 500]):
        non_nan = np.sum(~np.isnan(threshs[idx_thresh, :, 0]))
        theshs_non_nan = threshs[idx_thresh, :, 0][~np.isnan(threshs[idx_thresh, :, 0])]
        mode = stats.mode(theshs_non_nan)[0]
        mean = np.mean(theshs_non_nan)

        ax.axvline(mean, linestyle="--", color="red", label="Mean: "+str(int(np.ceil(mean)))+" (N="+str(non_nan)+")")
        ax.hist(theshs_non_nan, histtype="step", color="black", linestyle="-", lw=4, label=r'Histogram $\mathcal{L}(\mathbf{z})='+str(e[0])+r'\times 10^{'+str(e[1])+'}$')
        ax.legend(fontsize=22, loc=1)

        ax.set_xlim(0, m)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Iterations", fontsize=18)

    for ax in axarr:
        handles, labels = ax.get_legend_handles_labels()
        order = [1,0]
        ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], fontsize=20)

    axarr[0].set_ylabel("Number of Models", fontsize=20)
    for ax, label in zip(axarr, ["a)", "b)", "c)"]):
        ax.text(0, 102, label, fontsize=18)
