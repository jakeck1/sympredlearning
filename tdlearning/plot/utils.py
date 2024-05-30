import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def create_generalization_plot(steps_sym,steps_asym,stds=(None,None),save_path=None,overwrite=True,**plot_params):



    data = [steps_sym,steps_asym]
    agents = ['symmetric','asymmetric']


    colors = plot_params.get('colors',['red','blue'])
    figsize = plot_params.get('figsize',[12,5])
    title = plot_params.get('title','Generalization Performance of Agents')
    hlineheight  =plot_params.get('hline_height',170)
    lowerbound = plot_params.get('lower_bound',None)
    legend = plot_params.get('legend',False)
    #print(lowerbound)



    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel("Episode number",fontsize=14)
    ax.set_ylabel("Steps taken",fontsize=14)


    for i in range(2):
        agent = agents[i]

        episode_lens = data[i]

        mean = np.mean(episode_lens,0)
        if stds[i] is None:

            std = np.std(episode_lens,0)
        else:
            std = stds[i]


        lo = mean-std
        hi = mean+std
        print(lo.shape)
        print(hi.shape)
        color = colors[i]

        ax.plot(np.arange(len(mean)), mean, c=color, label=agent)
        ax.fill_between(np.arange(len(mean)), lo, hi, color=color, alpha=0.2)
    ax.axvline(episode_lens.shape[1]//2,0,hlineheight,linestyle='--',color='black')
    if lowerbound:
        ax.axhline(lowerbound, 0, len(mean), linestyle='--', color='grey',alpha=0.5)

    plt.title(title,fontsize=20)
    if legend:
        plt.legend()
    if save_path:
        if overwrite:
            plt.savefig(save_path)
    return fig,ax



def create_violin_plot(steps_sym,steps_asym,save_path=None,overwrite=True,ax = None,**plot_params):

    data = [steps_sym, steps_asym]
    agents = ['symmetric', 'asymmetric']

    colors = plot_params.get('colors', ['red', 'blue'])
    figsize = plot_params.get('figsize', [12, 5])
    title = plot_params.get('title', 'Generalization Performance of Agents')

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)


    col = np.hstack(data)
    types = np.hstack((['Symmetric' for i in range(len(data[0]))],['Asymmetric' for i in range(len(data[1]))]))


    df = {'Deviation from Minimal Path Length':col,'Agent':types}


    sns.set_context("paper")
    violins = sns.violinplot(df, x = 'Agent',y='Deviation from Minimal Path Length',palette=colors,density_norm='width',alpha=0.1,cut = 0.0,ax=ax)
    plt.setp(ax.collections, alpha=.5)
    ax.set_ylabel("Suboptimality",fontsize=14)
    
    violins.set_xticklabels(violins.get_xticklabels(), fontsize=14)
    ax.set_title(title,fontsize=20)
    if save_path:
        if overwrite:
            plt.savefig(save_path)
    return ax