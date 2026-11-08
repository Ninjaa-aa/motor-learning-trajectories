"""Auditable signal features for the recorded arc-steering task."""
import numpy as np
from scipy.signal import savgol_filter


def phase(day,block):
    if day==1 and block in (1,2,3):return 'pre'
    if day==1 and block in (95,96,97):return 'post'
    if day==2 and block in (0,1):return 'retention'
    return None


def segment(time,xy,settings):
    t=np.asarray(time,dtype=float);p=np.asarray(xy,dtype=float)
    if t.ndim!=1 or p.shape!=(len(t),2) or len(t)<settings['minimum_samples']:raise ValueError('insufficient_samples')
    if not np.isfinite(t).all() or not np.isfinite(p).all():raise ValueError('nonfinite_samples')
    if np.any(np.diff(t)<=0):raise ValueError('nonincreasing_timestamps')
    left=np.linalg.norm(p-settings['left_endpoint'],axis=1)<=settings['endpoint_radius_px']
    right=np.linalg.norm(p-settings['right_endpoint'],axis=1)<=settings['endpoint_radius_px']
    li=np.flatnonzero(left)
    if not len(li):raise ValueError('no_left_endpoint')
    ri=np.flatnonzero(right & (np.arange(len(t))>li[0]))
    if not len(ri):raise ValueError('no_right_endpoint_after_left')
    end=ri[0];start=np.flatnonzero(left & (np.arange(len(t))<end))[-1]
    t=t[start:end+1];p=p[start:end+1]
    if len(t)<settings['minimum_samples']:raise ValueError('short_segment')
    duration=t[-1]-t[0]
    if not settings['min_duration_seconds']<=duration<=settings['max_duration_seconds']:raise ValueError('duration_outside_bounds')
    if np.max(np.diff(t))>settings['max_gap_seconds']:raise ValueError('sampling_gap')
    return t-t[0],p,dict(start_index=int(start),end_index=int(end),raw_samples=len(time),segment_samples=len(t),max_gap_seconds=float(np.max(np.diff(t))))


def describe(time,xy,settings):
    t=np.asarray(time,dtype=float);p=np.asarray(xy,dtype=float)
    if len(t)<5 or p.shape!=(len(t),2) or not np.isfinite(p).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('invalid_signal')
    duration=t[-1]-t[0]
    count=max(21,round(duration*settings['resample_hz'])+1)
    grid=np.linspace(0,duration,count);dt=grid[1]-grid[0]
    uniform=np.column_stack([np.interp(grid,t-t[0],p[:,d]) for d in (0,1)])
    window=max(5,round(settings['savgol_window_seconds']/dt));window+=1-window%2
    window=min(window,count if count%2 else count-1)
    smooth=savgol_filter(uniform,window,settings['savgol_polynomial'],axis=0)
    jerk=savgol_filter(uniform,window,settings['savgol_polynomial'],deriv=3,delta=dt,axis=0)
    length=float(np.linalg.norm(np.diff(smooth,axis=0),axis=1).sum())
    delta=smooth[-1]-smooth[0];displacement=float(np.linalg.norm(delta))
    if min(length,displacement)<1e-6:raise ValueError('stationary_signal')
    dimensionless=duration**5/length**2*float(np.trapezoid(np.sum(jerk**2,axis=1),grid))
    ldj=-float(np.log(max(dimensionless,1e-15)))
    direction=delta/displacement
    rotation=np.array([[direction[0],-direction[1]],[direction[1],direction[0]]])
    aligned=(smooth-smooth[0])@rotation/displacement
    newgrid=np.linspace(0,duration,settings['shape_points'])
    shape=np.column_stack([np.interp(newgrid,grid,aligned[:,d]) for d in (0,1)])
    # shortcut: task-specific screen geometry; other tasks need their own calibrated endpoints and arc.
    radial=(np.linalg.norm(smooth-[500,500],axis=1)-400)/18
    return dict(duration=duration,ldj=ldj,path_ratio=length/displacement,radial_rmse=float(np.sqrt(np.mean(radial**2)))),shape
