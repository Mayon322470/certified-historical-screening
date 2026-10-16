"""Standalone scientific figures from reduced statistics, never raw certificates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_comparison import ROOT, MAIN_METHODS, POLICIES

LABELS = {'ma_strip':'Ma strip', 'ma_strip_structured':'Ma strip + structure',
          'zhao1':r'Zhao $\nu=1$', 'zhao2':r'Zhao $\nu=2$',
          'zhao2t3':r'Zhao $\nu=2$ + T3', 'zhao2t3_structured':r'Zhao $\nu=2$ + T3 + structure',
          'zhao2t3_sturm':'Zhao + T3 + Sturm only', 'zhao2t3_sparse':'Zhao + T3 + sparse only'}
COLORS = {'ma_strip':'#527DAD', 'ma_strip_structured':'#154E83', 'zhao1':'#B0A6CB',
          'zhao2':'#9586BB', 'zhao2t3':'#7561A8', 'zhao2t3_structured':'#4D357D',
          'zhao2t3_sturm':'#328C7B', 'zhao2t3_sparse':'#A97A29'}
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'axes.titlesize':12,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.facecolor':'white'})


def save(fig, folder, stem):
    paths = []
    for suffix in ('png', 'pdf'):
        path = folder/f'{stem}.{suffix}'
        fig.savefig(path, dpi=220, bbox_inches='tight')
        paths.append(str(path))
    plt.close(fig)
    return paths


def footer(report):
    if report['bootstrap']['enabled']:
        return ('Intervals: 10,000 case-bootstrap percentile 95% intervals; descriptive of this fixed suite. '
                'No population or repeat-level inference.')
    if 'method_sources' in report:
        return 'Single runs: full-structure main rep0; two ablations from a separate tag. No confidence intervals.'
    return 'Single selected run per method/case; no confidence intervals or repeat-level timing inference.'


def overview(report, folder, prefix):
    methods = [m for m in (*MAIN_METHODS, 'zhao2t3_sturm', 'zhao2t3_sparse') if m in report['methods']]
    vals = [report['methods'][m] for m in methods]
    y = np.arange(len(methods))
    fig, axes = plt.subplots(1,2,figsize=(12.2,5.4),gridspec_kw={'width_ratios':[1,1.25]})
    colors = [COLORS[m] for m in methods]
    axes[0].barh(y,[v['within_budget_all'] for v in vals],color=colors,height=.66)
    axes[0].set_yticks(y,[LABELS[m] for m in methods])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0,report['cases']*1.20)
    axes[0].set_xlabel('Cases resolved within budget')
    single = report['repeats_per_case_method'] == 1
    axes[0].set_title('A  Resolution in selected run' if single else 'A  Resolution in every repeat',loc='left',fontweight='bold')
    for yi,v in enumerate(vals):
        axes[0].text(v['within_budget_all']+.5,yi,f"{v['within_budget_all']}/{v['cases']}",va='center',fontsize=9)
    costs = [v['sum_case_median_seconds'] for v in vals]
    axes[1].barh(y,costs,color=colors,height=.66)
    axes[1].set_yticks(y,[])
    axes[1].invert_yaxis()
    ci_max = max([v['sum_case_median_seconds_ci95'][1] for v in vals if v['sum_case_median_seconds_ci95']] or costs)
    xmax = max(max(costs),ci_max)
    for yi,v in enumerate(vals):
        ci = v['sum_case_median_seconds_ci95']
        if ci:
            axes[1].plot(ci,[yi,yi],color='#252525',lw=1.15)
            axes[1].plot(ci,[yi,yi],linestyle='none',marker='|',color='#252525',ms=6)
        axes[1].text((ci[1] if ci else costs[yi])+.018*xmax,yi,f'{costs[yi]:.2f}',va='center',fontsize=9)
    axes[1].set_xlim(0,1.22*xmax)
    axes[1].set_xlabel('Total cost: sum of run seconds' if single else 'Total cost: sum of case-median seconds')
    axes[1].set_title('B  All-task wall-time cost',loc='left',fontweight='bold')
    for ax in axes:
        ax.grid(axis='x',color='#E6E8EB',linewidth=.6)
        ax.set_axisbelow(True)
    budget = report['run_metadata']['arguments']['budget']
    repeats_label = '1 selected run' if single else f"{report['repeats_per_case_method']} repeats"
    fig.suptitle(f"Fixed-suite comparison  |  {report['cases']} cases, {repeats_label}, {budget:g} s per run",
                 x=.03,y=.99,ha='left',fontsize=14,fontweight='bold')
    fig.text(.03,.062,'Cost includes UNKNOWN, late certificates, and errors; cost reduction is not solve speedup.',fontsize=9)
    fig.text(.03,.018,footer(report),fontsize=8,color='#50565E')
    fig.subplots_adjust(left=.25,right=.98,bottom=.21,top=.86,wspace=.2)
    return save(fig,folder,prefix+'_overview')


def ratio_label(pair):
    return LABELS[pair['baseline']]+'  →  '+LABELS[pair['candidate']]


def ratio_forest(report, folder, prefix):
    pairs = list(report['paired'].values())
    columns = [('All cases',None),('Ordinary cases','ordinary'),('Structured cases','structured')]
    fig,axes = plt.subplots(1,3,figsize=(17.0,max(5.5,.43*len(pairs)+2.2)),sharey=True)
    all_limits = [1.]
    for pair in pairs:
        for _,family in columns:
            p = pair if family is None else pair['families'].get(family,{})
            x = p.get('common_solved_geomean_speed_ratio')
            ci = p.get('common_solved_geomean_speed_ratio_ci95')
            if x is not None: all_limits.append(x)
            if ci: all_limits.extend(ci)
    low,high = np.log(min(all_limits)),np.log(max(all_limits))
    padding = max(.2,.13*(high-low))
    bounds = np.exp([low-padding,high+padding])
    for ax,(title,family) in zip(axes,columns):
        ax.axvline(1.,color='#888888',lw=1,ls='--')
        for y,pair in enumerate(pairs):
            p = pair if family is None else pair['families'].get(family,{})
            x = p.get('common_solved_geomean_speed_ratio')
            ci = p.get('common_solved_geomean_speed_ratio_ci95')
            color = '#155785' if pair['category']=='within_method' else '#786099' if pair['category']=='cross_paper' else '#337D70'
            if ci: ax.plot(ci,[y,y],color=color,lw=1.6)
            if x is not None: ax.plot(x,y,'o',color=color,ms=5)
            else: ax.text(1.,y,'no eligible cases',ha='center',va='center',fontsize=7.5,color='#6E7278',bbox={'fc':'white','ec':'none','pad':1})
            n = p.get('common_solved_nonquick_cases',0)
            slow = p.get('slower_by_over_5pct_count',0)
            ax.text(1.02,y,f'{n:2d} / {slow:2d}',transform=ax.get_yaxis_transform(),va='center',fontsize=8)
        ax.set_xscale('log')
        ax.set_xlim(bounds)
        ax.set_ylim(len(pairs)-.4,-.7)
        ax.set_title(title,loc='left',fontweight='bold',pad=20)
        ax.text(1.02,1.015,'n / slower',transform=ax.transAxes,fontsize=8,color='#50565E')
        ax.set_xlabel('Geometric mean time ratio')
        ax.grid(axis='x',which='both',color='#E6E8EB',lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(pairs)),[ratio_label(p) for p in pairs],fontsize=9)
    fig.suptitle('Paired solved-time ratios  |  baseline → candidate',x=.02,y=.985,ha='left',fontsize=14,fontweight='bold')
    fig.text(.02,.09,'Ratio = baseline / candidate; >1 favors candidate. n = eligible cases; slower = candidate >5% slower.',fontsize=9)
    eligibility_label = ('Eligible: selected runs resolve within budget, matching decisions, no conflicting certificates, and not common quick.'
                         if report['repeats_per_case_method'] == 1 else
                         'Eligible: all repeats resolve within budget, stable matching decisions, no conflicting certificates, and not common quick.')
    fig.text(.02,.056,eligibility_label,fontsize=9)
    fig.text(.02,.023,footer(report),fontsize=8,color='#50565E')
    fig.subplots_adjust(left=.33,right=.95,bottom=.21,top=.86,wspace=.38)
    return save(fig,folder,prefix+'_paired_ratios')


def threshold_resolution(report, folder, prefix):
    methods = [m for m in (*MAIN_METHODS,'zhao2t3_sturm','zhao2t3_sparse') if m in report['methods']]
    if not all(all(p in report['methods'][m]['threshold_policies'] for p in POLICIES) for m in methods):
        return []
    matrix = np.array([[report['methods'][m]['threshold_policies'][p]['within_budget_rate'] for p in POLICIES] for m in methods])
    fig,ax = plt.subplots(figsize=(9.2,5.0))
    im = ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues',aspect='auto')
    ax.set_yticks(range(len(methods)),[LABELS[m] for m in methods])
    ax.set_xticks(range(4),['Public bound B\n+5% scale','Numerical M\n+1% scale','Numerical M\n+0.01% scale','Numerical M\n−0.1% scale'])
    for i,m in enumerate(methods):
        for j,p in enumerate(POLICIES):
            v = report['methods'][m]['threshold_policies'][p]
            ax.text(j,i,f"{v['within_budget_all']}/{v['cases']}",ha='center',va='center',fontsize=11,
                    color='white' if matrix[i,j]>.6 else '#203040')
    for spine in ax.spines.values(): spine.set_visible(False)
    resolution_label = 'Fraction resolved in selected run' if report['repeats_per_case_method'] == 1 else 'Fraction resolved in every repeat'
    fig.colorbar(im,ax=ax,pad=.035,label=resolution_label)
    ax.set_title('Within-budget resolution by predeclared threshold policy',loc='left',fontweight='bold',pad=17)
    fig.text(.025,.055,'Entries show case counts. M is an uncertified numerical reference; signed margins are difficulty labels, not safety truth.',fontsize=8.5)
    fig.text(.025,.018,'All cases are retained, including UNKNOWN, errors, and certificates obtained after the time budget.',fontsize=8.5,color='#50565E')
    fig.subplots_adjust(left=.31,right=.94,bottom=.25,top=.86)
    return save(fig,folder,prefix+'_threshold_resolution')


def make(tag, summary='paired_summary.json', prefix='comparison', root=ROOT):
    folder = Path(root)/'results'/tag
    report = json.loads((folder/summary).read_text(encoding='utf-8'))
    if not report['integrity']['complete']:
        raise ValueError('Refusing to draw final figures for an incomplete run')
    paths = overview(report,folder,prefix)
    if report['paired']: paths.extend(ratio_forest(report,folder,prefix))
    paths.extend(threshold_resolution(report,folder,prefix))
    print(json.dumps(dict(figures=paths),indent=2))
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('--summary',default='paired_summary.json')
    parser.add_argument('--prefix',default='comparison')
    args = parser.parse_args()
    make(args.tag,args.summary,args.prefix)
