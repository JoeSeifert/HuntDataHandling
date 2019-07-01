# -*- coding: utf-8 -*-

import os, re
from itertools import zip_longest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

# Update from old_lib.py on 5/2/2019
# Finalized changes on 5/23/2019

# Some ideas:


@dataclass
class Config:
    # in current form, could just use kwargs?
    '''
     config works like in Matlab
     config1 = Config()
     config1.B = 2
     config1.Vg = 5
     etc.
     can do config1._from_list(('iterable', 'with', 'labels', 'in', 'correct', 'order'))
     can do config1 = Config(B=2, Vg=3, T=1) (<-- I like this)
     '''

    def __init__(self, **kwargs):
        self._from_dict(kwargs)
        self.labels = self._get_df_labels()
        
    def edit(self, label, index):
        self.__dict__[label] = index
        self.labels = self._get_df_labels()

    def _from_list(self, label_list):
        # labels in correct order
        # insert None where you don't want any label?
        for i, label in enumerate(label_list):
            if label is not None:
                self.__dict__[label] = i

    def _from_dict(self, label_dict):
        # label_dict is of form {str label: int loc}
        for label, loc in label_dict.items():
            if label is not None:
                self.__dict__[label] = loc

    def _get_df_labels(self):
        # returns ordered list of column labels, None where label is not specified
        cols = [elem for elem in dir(self) if type(getattr(self, elem)) is int]  # gets only labels I assigned
        current = 0
        final = []
        while True:
            found = False
            for i, elem in enumerate(cols):
                if getattr(self, elem) == current:
                    final.append(cols.pop(i))
                    found = True
            if not found:
                final.append(None)
            if len(cols) == 0:
                break
            current += 1
        return final
    
class Measurement:
    
    def __init__(self, fnums, config=None, path=None):
        self.fnums = fnums
        self.path = path # path or subfolder to data files. None = current folder
        self.config = config if config is not None else Config()
        self._re_dict = {
            'full_path': re.compile(r'[CDEFGHI]:\\.+?'), # i.e. starts with hard drive name
            'unit': re.compile(r'\([a-zA-Z]+\)'),
            'fnum': re.compile(r'0{0,4}\d{1,4}_')  # for finding fnum when it's not specified
        }
        
        self.raw_df = pd.DataFrame()
        self.units = {} 
        self._import_data()
    
    def _import_data(self):
        '''
        Import all data into self.raw_df
        -fnums = iterable containing integer file numbers to import
        -path = full path or subfolder of current directory where data is stored
        '''
        #CHANGES:
        #   -make 'time' column a time series in pandas
        
        d = os.getcwd()
        full_path_re = self._re_dict['full_path']
        find_fnum_re = self._re_dict['fnum']
        unit_re = self._re_dict['unit']
        
        if self.path is None:  # data in same folder as analysis file
            data_loc = d
        elif not full_path_re.match(self.path):  # data in subfolder specfied
            data_loc = f'{d}\\{self.path}'
        else:  # data_loc is a full path to data file
            data_loc = self.path
        
        if (self.fnums is not None) and not (hasattr(self.fnums, '__iter__')):
            self.fnums = [int(self.fnums)]
            
        files = [fn for fn in os.listdir(data_loc) if fn.endswith('.dat')]
        
        dfs = []
        if self.fnums is None:
            for fn in files:
                temp = pd.read_csv(f'{data_loc}\\{fn}', sep='\t', header=0)
                if not temp.empty:
                    m = find_fnum_re.match(fn)
                    temp['scan'] = int(m.group()[:-1])  # make scan num column
                    dfs.append(temp)
        else:
            for n in self.fnums:
                fnum_re = re.compile(f'0{{0,4}}{n}_')  # for finding a specific fnum
                fn = None
                for file in files:
                    if fnum_re.match(file):
                        fn = file
                if fn is None:
                    print(f'fnum {n} not found')
                    continue
                temp = pd.read_csv(f'{data_loc}\\{fn}', sep='\t', header=0)
                if not temp.empty:
                    temp['scan'] = n
                    dfs.append(temp)
        self.raw_df = pd.concat(dfs, sort=False)
        
        # infer units, rename columns
        new_labels = {}
        for c_label, df_label in zip_longest(self.config.labels, self.raw_df.columns):
            label = c_label if c_label is not None else df_label
            unit = unit_re.findall(df_label)
            if len(unit) == 1:
                label = label.replace(unit[0], '')
                self.units[label] = unit[0][1:-1]
            new_labels[df_label] = label
        self.raw_df.rename(new_labels, axis='columns', inplace=True)
        
        # strip spaces from column names
        self.raw_df.rename(str.strip, axis='columns', inplace=True)
            
        #put scan column at the end
        new_cols = [col for col in self.raw_df.columns if col != 'scan']
        new_cols.append('scan')
        self.raw_df = self.raw_df[new_cols]
        
    def interp(self, x_ax, y_ax, z_ax, n_points=201, as_df=False, nx=None, ny=None, grids=(None, None)):
        '''
        Interpolate 2D data over a uniform mesh, lineraly.
        Input should be like:
                ms.interp(c.T, c.Vg, c.Rxx) (c = Config object)
                ms.interp('T', 'Vg', 'Rxx') ('T' etc. are column labels)
        
        :param x_ax, y_ax, z_ax: DataFrame label or column number to put on x/y/z-axis
        :param n_points: Number of grid points on a side for interpolation. 
            Default= 201, square grid. Non-square grid not supported
        :param as_df: bool. Default False.
            True --> return a tidy pandas DataFrame. False --> return grids x, y, z.
        :return: grid_x, grid_y, grid_z
            grid_x, grid_y = grid of (x, y) coordinates
            grid_z = interpolated data corresponding to (x, y)
        '''
        if type(x_ax) is int: # i.e. "df = ms.interp(c.T, c.Vg, c.Rxx)"
            x_ax = self.raw_df.columns[x_ax]
            y_ax = self.raw_df.columns[y_ax]
            z_ax = self.raw_df.columns[z_ax]
            
        nx = n_points if nx is None else None
        ny = n_points if ny is None else None
        
        grid_x, grid_y = grids
        
        if grid_x is None and grid_y is None:
            grid_x, grid_y = np.meshgrid(np.linspace(self.raw_df[x_ax].min(), self.raw_df[x_ax].max(), nx),
                                         np.linspace(self.raw_df[y_ax].min(), self.raw_df[y_ax].max(), ny))
        
        # Let's try normalizing our x & y axes
        
        def norm(arr, to=None):
            if to is None: to = arr
            return (arr - arr.min()) / arr.max()
        
        raw_x = self.raw_df[x_ax].values
        raw_y = self.raw_df[y_ax].values
        norm_grid_x = norm(grid_x, to=raw_x)
        norm_grid_y = norm(grid_y, to=raw_y)
        
        points = np.array(list(zip(norm(raw_x), norm(raw_y)))) # notice normalized x and y
        values = self.raw_df[z_ax].values
        
        grid_z = griddata(points, values, (norm_grid_x, norm_grid_y), method='linear')
        if as_df:
            return pd.DataFrame({x_ax: grid_x.ravel(), y_ax: grid_y.ravel(), z_ax:grid_z.ravel()})
        else:
            return grid_x, grid_y, grid_z

class MeasurementLibrary:
    
    # CURRENT MAJOR ISSUES:
    #
    # CURRENT MINOR ISSUES:
    #   -It could group 'similar' measurements a bit better.
    #       -Measurements with distince froots, but very similar aside from one change
    #       -ex: field-sweeps, QH configs
    # THINGS TO ADD:
    #
    #   -'1kOhm' or 'XkOhm' keyword in title -- means current is a voltage measured through X kOhm resistor
    
    def __init__(self, path=None):
        self._data_loc = self._get_data_loc(path)
        
        # self._re_dict --> key = searchable quantity, value regex object
        self._re_dict = {
            'fnum': re.compile(r'0{0,4}\d{1,4}_'),
            'config': re.compile(r'\d{1,2}-\d{1,2}-\d{1,2}-\d{1,2}'),
            'config2t': re.compile(r'2t-\d{1,2}-\d{1,2}'),
            'B': re.compile(r'(-?\d+(?:\.\d+)?)(oe|Oe|T)'),
            'I': re.compile(r'(-?\d+(?:\.\d+)?)((p|n|u|m)A)'),
            'Vg': re.compile(r'(-?\d+(?:\.\d+)?)(V)(?:Vg)?'),
            'T': re.compile(r'(\d+(?:\.\d+)?)(m?K)'),
            'freq': re.compile(r'(\d+(?:\.\d+)?)((k|m|M|g|G){0,1}(h|H)z)'),
        }
        
        # self._units_dict --> key = unwanted unit, value = conversion factor
        # (unwanted quantity) * (conversion factor) = (correct quantity)
        self._units_dict = {
            'oe': 1.e-4,
            'pa': 1.e-12,
            'na': 1.e-9, 
            'ua': 1.e-6, 
            'ma': 1.e-3,
            'khz': 1.e3,
            'mhz': 1.e6,
            'ghz': 1.e9,
            'mk': 1e-3,
        }
        
        # make DataFrames
        self.lib = self._make_library()
        self.lib = pd.merge(self.lib, self._make_params(), how='outer')
        
        self.info = self.lib[['meas_id', 'fstart', 'fend', 'froot']]
        self.params = self.lib[['B', 'Vg', 'I', 'T', 'freq', 'config1', 'config2', 'misc']]
        
    def __getitem__(self, i):
        return self.lib[i]
            
    def _get_data_loc(self, path):
        d = os.getcwd()
        full_path_re = r'[CDEFGHI]:\\.+?'  # i.e. starts with hard drive name
        if path is None:  # data in same folder as analysis file
            return d
        elif not re.match(full_path_re, path):  # data in subfolder specfied
            return f'{d}\\{path}'
        else:  # data_loc is a full path to data file
            return path
        
    def _get_configs(self, froot):
        config_re = self._re_dict['config']
        config_re_2t = self._re_dict['config2t']
        
        # gather configs, put them in order
        configs = []
        for m in config_re.finditer(froot):
            configs.append((m.span()[0], m.group(0)))  # m.span() = location in string
        for m in config_re_2t.finditer(froot):
            configs.append((m.span()[0], m.group(0)))
			
        configs.sort()  # by first elem in tuple, i.e. location in string
        configs = [c[1] for c in configs] # remove location
        while len(configs) < 2:
            configs.append(None)
        return configs
    
    def _get_fnum(self, fname):
        fnum_re = self._re_dict['fnum']
        fnum = fnum_re.match(fname).group(0)
        return int(fnum[:-1])  # remove '_', make it a number
        
    def _make_library(self):
        # Make list of tuples of fnum and filenames: (fnum, fname), and sort
        files = [(self._get_fnum(fname), fname) 
                    for fname in os.listdir(self._data_loc) 
                    if fname.endswith('.dat')]
        files.sort()  # sorts by first element of tuple (fnum)
        
        # Initialize some variables:
        rows = []
        row_dict = {}
        current_froot = None
        meas_id = 0
        fstart = None
        fend = None
        config1, config2 = None, None
        
        for fnum, fname in files:
            # remove fnum from fname
            fname = fname.replace(str(fnum).rjust(3, '0')+'_', '')
            froot = fname[:-4]  # remove '.dat'
            
            # group the measurements
            if current_froot is None: # for the first file
                fstart = fnum
                fend = fnum
                current_froot = froot
                config1, config2 = self._get_configs(froot)
            elif froot == current_froot: # for middle files in a megasweep
                fend = fnum
            else: # for beginning files in different measurements (except the first measurement)
                row_dict = {'meas_id': meas_id, 'fstart': fstart, 'fend': fend,
                            'froot': current_froot, 'config1': config1, 'config2': config2}
                rows.append(row_dict)
                meas_id += 1
                fstart = fnum
                fend = fnum
                current_froot = froot
                config1, config2 = self._get_configs(froot)
        
        # add the very last file/measurement to the library
        row_dict = {'meas_id': meas_id, 'fstart': fstart, 'fend': fend,
                    'froot': froot, 'config1': config1, 'config2': config2}
        rows.append(row_dict)
        
        # make the library
        return pd.DataFrame(rows, columns=('meas_id', 'fstart', 'fend',
                                           'froot', 'config1', 'config2'))
        
    def _make_params(self):
        quantities = ('B', 'I', 'Vg', 'T', 'freq')
        rows = []
        for meas_id, froot in zip(self.lib['meas_id'].values, self.lib['froot'].values):
            row_dict = {'meas_id': meas_id}
            misc = []
            
            # search for the relevant quantities
            for q in quantities:
                re_obj = self._re_dict[q]  # grab appropraite regex object
                m = re_obj.search(froot)
                if m:
                    try:
                        # fix units if you need to
                        # units_dict gives a conversion factor to the correct units
                        value =  float(m.group(1)) * self._units_dict[m.group(2).lower()]
                    except KeyError:
                        value = float(m.group(1))
                    row_dict[q] = value    
                    froot = froot.replace(m.group(0) + '_', '')
                else:
                    row_dict[q] = None
                    
            # Store everything else in 'misc' (except configs)
            for elem in froot.split('_'):
                config_re = self._re_dict['config']  # grab correct re_obj
                config_re_2t = self._re_dict['config2t']
                for re_obj in (config_re, config_re_2t):
                    m = re_obj.search(elem)
                    if m is not None:  # loops breaks when it finds a config
                        break
                if m is None:  # m is still same value as previous loop
                    misc.append(elem)            
            row_dict['misc'] = '_'.join(misc)
            
            # add to rows
            rows.append(row_dict)
            
        # make the DataFrame
        return pd.DataFrame(rows, columns=('meas_id', *quantities, 'misc'))
    
    def _get_meas_by_id(self, meas_id, config=None):
        if type(meas_id) is int: # single meas_id given
            df = self.lib[self.lib['meas_id'] == meas_id]
            fstart, fend = df['fstart'].values, df['fend'].values
            yield Measurement(np.arange(fstart, fend + 1), config=config, 
                              path=self._data_loc), df
        else: # list of meas_id's given
            for elem in meas_id:
                df = self.lib[self.lib['meas_id'] == elem]
                fstart, fend = df['fstart'].values, df['fend'].values
                yield Measurement(np.arange(fstart, fend + 1), config=config,
                                  path=self._data_loc), df
  
    def _get_meas_by_df(self, df, config=None):
        for ind, row in df.iterrows():
            fstart, fend = row['fstart'], row['fend']
            yield Measurement(np.arange(fstart, fend + 1), config=config,
                              path=self._data_loc), df[df['meas_id'] == row['meas_id']]
        
    def get_measurement(self, meas_id=None, df=None, config=None, **kwargs):
        '''
        Return a generator using either:
            -a) a list of measurement ID's 
            -b) a DataFrame with rows in order to iterate
        Generator returns Measurement objects for each specified measurement.
        '''
        if meas_id is not None:
            return self._get_meas_by_id(meas_id, config=config)
        elif df is not None:
            return self._get_meas_by_df(df, config=config)

##############
# SOME TOOLS #
##############
        
def grid_to_df(grid_x, grid_y, grid_z, labels=None):
    # interpolation grid output to interpolation df output
    # MUST RENAME AXES
    if labels is None:
        return pd.DataFrame({'1': grid_x.ravel(), '2': grid_y.ravel(), '3':grid_z.ravel()})
    else:
        return pd.DataFrame({labels[0]: grid_x.ravel(), labels[1]: grid_y.ravel(), labels[2]:grid_z.ravel()})

def df_to_grid(df, x_ax, y_ax, z_ax):
    # interpolation df output to interpolation grid output
    grid_x, grid_y = np.meshgrid(df[x_ax].unique(), df[y_ax].unique())
    grid_z = df[z_ax].values.reshape(grid_x.shape)
    
    return grid_x, grid_y, grid_z

def linecut(x, y, z, direc, value):
    '''
    WORKS ON INTERPOLATED DATA
    x, y, z = np arrays with x, y, z values respectively
    direc = 'h' or 'v', horizontal or vertical
    value = value at which to take a linecut
    '''
    
    if direc=='h':
        ind = np.where(abs(y[:,0] - value) == abs(y[:,0] - value).min())[0][0]
        act_value = y[ind, 0]
        if act_value != value:
            print(f'Closest matching value = {act_value}. Line cut taken here.')
        return x[ind, :], z[ind, :]
    elif direc=='v':
        ind = np.where(abs(x[0,:] - value) == abs(x[0,:] - value).min())[0][0]
        act_value = x[0, ind]
        if act_value != value:
            print(f'Closest matching value = {act_value}. Line cut taken here.')
        return y[:, ind], z[:, ind]
    else:
        print(f"Direction must be either 'h' or 'v', not {direc}")
        
def boxcar(line, x_pts=1):
    # averages all points within x_pts of the i-th position of line.
    # "box-car average"
    newline = []
    for pos, val in enumerate(line):
        temp = []
        for i in range(pos - x_pts, pos + x_pts + 1):
            if 0 <= i < len(line):
                temp.append(line[i])
        newline.append(sum(temp) / len(temp))
    return np.array(newline)

def find_peak(x, y, plot=False):
    return find_feature(x, y, plot=plot, mode='peak')

def find_trough(x, y, plot=False):
    return find_feature(x, y, plot=plot, mode='trough')

def find_feature(x, y, plot=False, mode='peak'):
    # x, y = np arrays corresponding to x, y data with a peak
    
    def gauss(x, amp, x0, s, off, a):
        if mode == 'peak':
            return (amp * np.exp(-((x - x0) / s)**2) + off) + (a*(x - x0))
        elif mode == 'trough':
            return (-amp * np.exp(-((x - x0) / s)**2) + off) + (a*(x - x0))
    
    # look for peak around maximum
    if mode == 'peak': 
        center_ind = np.argmax(y)
    elif mode == 'trough':
        center_ind = np.argmin(y)
    center_x = x[center_ind]
    
    # choose sample points around peak
    n_points = int(x.shape[0] / 6)  # arbitrary as fuuuuck
    
    start_ind, end_ind = center_ind - n_points, center_ind + n_points
    if start_ind < 0:
        start_ind = 0
    if end_ind > x.shape[0]:
        end_ind = x.shape[0]
        
    samp_x = x[start_ind:end_ind + 1]
    samp_y = y[start_ind:end_ind + 1]
    
    # guess at slope of background line
    m = (samp_y[-1] - samp_y[0]) / (samp_x[-1] - samp_x[0])
    
    # fit, with guess params
    guess = (samp_y.max(), center_x, 1, samp_y.min(), m)
    popt, _ = curve_fit(gauss, samp_x, samp_y, p0=guess)
    g = gauss(samp_x, *popt)
    
    if mode == 'peak':
        center = (samp_x[np.argmax(g)], np.max(g))
    elif mode == 'trough':
        center = (samp_x[np.argmin(g)], np.min(g))
    
    # maybe plot it
    if plot:
        plt.plot(x, y, '-', label='Data', zorder=-1)
        plt.plot(samp_x, samp_y, 'r.', label='Sample', zorder=-1)
        plt.plot(samp_x, g, label='Fit', zorder=-1)
        plt.scatter(center[0], center[1], color='green', s=50, label='Max', zorder=1)
        plt.legend()
        plt.show()
        
    # calculate R^2
    resid = samp_y - g
    ss_resid = np.sum(resid**2)
    mean_y = np.mean(samp_y)
    ss_tot = np.sum((samp_y - mean_y)**2)
    r_sq = 1 - (ss_resid / ss_tot)
    
    # voila
    return center, (r_sq, popt)
    
    
    
    
    
    
    
