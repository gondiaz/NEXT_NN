import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def bounding_box(hits : pd.DataFrame, bounding_box_centre) :
    """Returns two arrays defining the coordinates of a box that bounds the voxels"""
    posns = hits[['X','Y','Z']]
    mins  = np.min(posns)
    maxs  = np.max(posns)
    max_dist = np.max([np.abs(bounding_box_centre-mins),np.abs(bounding_box_centre-maxs)], axis=0)
    return bounding_box_centre - max_dist, bounding_box_centre + max_dist

def voxelize_hits_per_ev(hits             : pd.DataFrame,
                         voxel_dimensions : np.ndarray,
                         strict_voxel_size: bool = False) -> pd.DataFrame:
    if len(hits)==0:
        raise NoHits
    bounding_box_centre = np.sum(hits[['X','Y','Z']].transpose()*hits['E']/np.sum(hits['E']), axis=1)
    hlo, hhi = bounding_box(hits, bounding_box_centre)
    bounding_box_size   =  hhi - hlo
    number_of_voxels = np.ceil(bounding_box_size / voxel_dimensions).astype(int)
    number_of_voxels = np.clip(number_of_voxels, a_min=1, a_max=None)
    if strict_voxel_size: half_range = number_of_voxels * voxel_dimensions / 2
    else                : half_range =          bounding_box_size          / 2
    voxel_edges_lo = bounding_box_centre - half_range
    voxel_edges_hi = bounding_box_centre + half_range
    eps = 3e-12 # geometric mean of range that seems to work
    voxel_edges_lo -= eps
    voxel_edges_hi += eps
    hit_positions = np.array(hits[['X','Y','Z']], dtype='float64')
    hit_energies  =          hits['E']
    E, edges = np.histogramdd(hit_positions,
                              bins    = number_of_voxels,
                              range   = tuple(zip(voxel_edges_lo, voxel_edges_hi)),
                              weights = hit_energies)    
    return E

def fit_data_to_size(data):
    pad_b  = np.max([[0,0,0],((size-np.array(data.shape))/2)]).astype('int')
    pad_a  =  np.max([[0,0,0],(size-np.array(data.shape))-pad_b]).astype('int')
    padded = np.pad(data,np.column_stack([pad_b,pad_a]), mode='constant')
    crop_b  = ((np.array(padded.shape)-size)/2).astype('int')
    crop_a  =  size+crop_b    
    cropped= padded[crop_b[0]:crop_a[0],crop_b[1]:crop_a[1],crop_b[2]:crop_a[2]]
    return cropped

def make_data_voxels(infile, voxel_dimension, size, threshold=0.8):
    df=pd.read_hdf(infile)
    datas=[];ens=[];evs=[]; labs=[];ev_rej=[]; en_true=[]
    for ev, dfp in df.groupby('event'):
        data = voxelize_hits_per_ev(dfp, voxel_dimension)
        ens_full = data.sum()
        data = fit_data_to_size(data,size)
        en_after_frame=data.sum()
        if (threshold*ens_full>en_after_frame):
            ev_rej.append(ev)
        else:
            ens.append(np.array([ens_full,en_after_frame]))
            datas.append(data/en_after_frame)
            evs.append(ev)
            try:
                labs.append(dfp['label'].unique()[0])
                en_true.append(dfp['true_energy'].unique()[0])
            except KeyError:
                labs.append(-1)
                en_true.append(-1)
                
    datas   = np.array(datas)
    labs    = np.array(labs)
    ens     = np.array(ens)
    en_true = np.array(en_true)
    return datas, labs, en_true,evs, ens, ev_rej


def plot_3d_vox(xarr, th=0,normalize=True, color=None,edgecolor=None, labels=False):
    dim=xarr.shape
    x=range(xarr.shape[0])
    y=range(xarr.shape[1])
    z=range(xarr.shape[2])
    fig = plt.figure(figsize=[12,10])
    voxels=(xarr>th)
    norm = mpl.colors.Normalize(vmin=xarr.min(), vmax=xarr.max())
    cmap = cm.jet
    m =cm.ScalarMappable(norm=norm, cmap=cmap)
    if labels:
        to_col={1:'red',2:'blue',3:'green'}
        colors=np.empty((xarr.shape[0],xarr.shape[1],xarr.shape[2]),dtype='object')
        nonzero=np.nonzero(xarr)
        for i,j,k in zip(*nonzero):
            colors[i,j,k]=to_col[xarr[i,j,k]]
        ax = fig.add_axes([0.,0,0.9, 1.],projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor=edgecolor)
    else:
        colors=np.empty(((xarr.shape[0],xarr.shape[1],xarr.shape[2],4)))
        nonzero=np.nonzero(xarr)
        for i,j,k in zip(*nonzero):
            colors[i,j,k]=m.to_rgba(xarr[i,j,k])
        ax = fig.add_axes([0.,0,0.9, 1.],projection='3d')
        ax.voxels(voxels, facecolors=colors, edgecolor=edgecolor)
        ax2 = fig.add_axes([0.9,0.,0.04, 1])

        cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')

    ax.set_xlim(0, dim[0])
    ax.set_ylim(0, dim[1])
    ax.set_zlim(0, dim[2])
    plt.show()



def make_equal_sample_and_shuffle(x, y, event):
    mask_y=(y>0)
    y_signal=y[mask_y]
    x_signal=x[mask_y]
    y_back=y[~mask_y]
    x_back=x[~mask_y]
    x_eq=np.concatenate((x_signal, x_back[:len(x_signal)]), axis=0)
    y_eq=np.concatenate((y_signal, y_back[:len(x_signal)]), axis=0)
    events_eq=np.concatenate((event[mask_y], event[~mask_y][:len(x_signal)]), axis=0)
    s=np.arange(len(x_eq))
    np.random.seed(0)
    x_eq=x_eq[s]
    y_eq=y_eq[s]
    evs_eq=evs_eq[s]
    return x_eq,y_eq,evs_eq
