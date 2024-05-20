import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import FancyArrowPatch

def antidiagonal_to_nan(arr):
    n = arr.shape[0]
    for i in range(n):
        arr[i, n - 1 - i] = np.nan
def add_cartesian_arrows(axes, arrow_length_ratio=0.02):
    # Calculate the x and y arrow sizes based on the axes' limits and the desired length ratio
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    x_arrow_length = (xlim[1] - xlim[0]) * arrow_length_ratio
    y_arrow_length = (ylim[1] - ylim[0]) * arrow_length_ratio

    # Create and add the arrow patches to the axes
    axes.add_patch(FancyArrowPatch((xlim[0], 0), (xlim[1]-0.1 + x_arrow_length, 0),
                                   mutation_scale=20, color='black', arrowstyle='->', linewidth=1))
    axes.add_patch(FancyArrowPatch((0, ylim[0]), (0, ylim[1]-0.1 + y_arrow_length),
                                   mutation_scale=20, color='black', arrowstyle='->', linewidth=1))

    # Remove the original axes lines
    axes.spines['left'].set_color('none')
    axes.spines['bottom'].set_color('none')
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    axes.xaxis.set_ticks_position('none')
    axes.yaxis.set_ticks_position('none')
    axes.xaxis.set_tick_params(labelbottom=False)
    axes.yaxis.set_tick_params(labelleft=False)

def create_convergence_plot(convergence_Ws,convergence_Vs,title=None,**exp_params):

    dt = exp_params.get('dt',0.01)

    fig, ax = plt.subplots(figsize=[10, 5])
    t = np.arange(convergence_Ws.shape[1]) * dt
    for i in range(convergence_Ws.shape[0]):
        ax.plot(t, convergence_Ws[i] * 1e3, alpha=0.2, color='red')
        ax.plot(t, convergence_Vs[i] * 1e3, alpha=0.2, color='blue')
    ax.plot(t, np.mean(convergence_Ws * 1e3, axis=0), alpha=1.0, color='red', label=r'$W^r$')
    ax.plot(t, np.mean(convergence_Vs * 1e3, axis=0), alpha=1.0, color='blue', label=r'$W^f$')
    #ax.set_xlim(0, 1000)
    ax.set_xlabel(r'$t[s]$')
    ax.set_ylabel('Distance to equilibrium [a.u.]', fontsize=20)
    plt.legend()

    if title:
        plt.title(title,fontsize=32)

    return fig


def create_walk_plots(states):


    fig, ax = plt.subplots()
    plt.rcParams.update({'axes.labelsize': 60})

    ax.plot(states, 'p', color='blue', markersize=0.5)

    ax.set_xlabel('time')

    ax.set_ylabel(r'$\theta$')

    ax.set_yticks([])
    ax.set_xticks([])
    add_cartesian_arrows(ax)
    return fig


def create_circular_rw_plots(SRs):




    SR_1,SR_2 = SRs
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title(r'$SR_{p^1}$')
    ax[1].set_title(r'$SR_{p^2}$')
    ax[0].matshow(SR_1.mean(axis=0), cmap='magma')
    ax[1].matshow(SR_2.mean(axis=0), cmap='magma')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])


    return fig



def plot_convergence_matrix(MW,parameters,max_value=None,logscale=False,nan_on_antidiagonal=False,title = None):


    MW = MW.copy()
    if max_value:
        MW[MW>max_value]=np.nan
    if logscale:
         MW = np.log(MW+1e-16)

    MW = np.nanmean(MW,axis=0)
    if nan_on_antidiagonal:
        antidiagonal_to_nan(MW)


    fig,ax = plt.subplots(figsize=[12,10])
   
    cax = ax.matshow(MW)
    ax.set_xticks(np.arange(len(parameters)))
    ax.set_yticks(np.arange(len(parameters)))
    ax.set_xlabel(r'$\alpha$',fontsize=16)
    ax.set_ylabel(r'$\beta$',fontsize=16)
    ax.set_xticklabels(np.round(parameters,2))
    ax.set_yticklabels(np.round(parameters,2))
   # fig.colorbar(cax)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = plt.colorbar(cax, cax=cbar_ax)
    # fig.colorbar(cax)
    if logscale:
            colorbar.set_label(r'$\ln{\frac{\mathcal{L}(W,W^*)}{\mathcal{L}(W_0,W^*)}}$',fontsize=20)

    else:
        colorbar.set_label(r'$\frac{\mathcal{L}(W,W^*)}{\mathcal{L}(W_0,W^*)}$',fontsize=20)

    if title is None:
         title = r'Convergence for different $\alpha,\beta$'
    plt.suptitle(title,fontsize=24)

    return fig

def histogram_plot(data, bins, ax, color, **kwargs):

    weights = np.ones_like(data) / len(data)

    ax.hist(data, bins=bins, weights=weights * 100, color=color, alpha=0.5, **kwargs)

def get_diffs(s, d=5):
    smooth_first_laps = s[:, :d]
    smooth_last_laps = s[:, -d:]

    diffs = smooth_last_laps - smooth_first_laps
    return diffs

def create_shift_experiment_plots(s_1,s_2):




    plt.rcParams.update({'font.size': 12})

    n_cells_layer_1 = s_1.shape[-1]
    n_cells_layer_2 = s_2.shape[-1]



    s_1_mean = np.mean(s_1, axis=(0, 2))
    s_1_std = np.std(s_1, axis=(2)).mean(axis=0)
    s_1_sem = s_1_std / (np.sqrt(n_cells_layer_1))

    s_2_mean = np.mean(s_2, axis=(0, 2))
    s_2_std = np.std(s_2, axis=(2)).mean(axis=0)
    s_2_sem = s_2_std / (np.sqrt(n_cells_layer_2))



    fig,ax = plt.subplots(ncols = 3,figsize=[15,5])


    ax[2].errorbar(np.arange(len(s_1_mean)), s_1_mean, yerr=s_1_sem, fmt='o', color='red', ecolor='red', elinewidth=1,
                 alpha=0.5,capsize=3)
    ax[2].errorbar(range(len(s_2_mean)), s_2_mean, yerr=s_2_sem, fmt='o', color='blue', ecolor='blue', elinewidth=1,
                 alpha=0.5,capsize=3)


    ax[2].set_ylabel('population shift relative to lap 12 [cm]')
    ax[2].set_xlabel('Lap #')


    diffs_1 = get_diffs(s_1)
    diffs_2 = get_diffs(s_2)

    bins = np.linspace(-2.4, 2.4, 40)

    histogram_plot(diffs_1.flatten()/24,bins,ax[0],'red')
    histogram_plot(diffs_2.flatten()/24,bins,ax[1],'blue')
    ax[0].set_ylim(0,45)
    ax[1].set_ylim(0,45)
    ax[0].set_xlabel('Place field shift distance (cm/lap)')
    ax[1].set_xlabel('Place field shift distance (cm/lap)')
    ax[0].set_ylabel('Frequency (%)')
    ax[1].set_ylabel('Frequency (%)')


    return fig,ax
