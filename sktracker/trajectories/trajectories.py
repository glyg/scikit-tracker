# -*- coding: utf-8 -*-


from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import numpy as np
import pandas as pd
import scipy as sp

from pandas.io import pytables

from .measures.transformation import time_interpolate as time_interpolate_
from .measures.transformation import transformations_matrix
from ..utils import print_progress

import logging
log = logging.getLogger(__name__)


__all__ = []


class Trajectories(pd.DataFrame):
    """
    This class is a subclass of the class :class:`pandas.DataFrame`

    It is mainly here to provide utility attributes and syntactic shugar.

    Attributes
    ----------
    t_stamps : ndarray
        unique values of the `t_stamps` index of `self.trajs`

    labels : ndarray
        unique values of the `labels` index of `self.trajs`

    iter_segments : iterator
        yields a `(label, segment)` pair where `label` is iterated over `self.labels`
        and `segment` is a chunk of `self.trajs`

    segment_idxs : dictionnary
        Keys are the segent label and values are a list
        of  `(t_stamp, label)` tuples for each time point of the segment

    Parameters
    ----------
    trajs : :class:`pandas.DataFrame`

    """
    def __init__(self, *args, **kwargs):
        """
        """
        super(Trajectories, self).__init__(*args, **kwargs)

    @classmethod
    def empty_trajs(cls, columns=['x', 'y', 'z']):
        empty_index = pd.MultiIndex.from_arrays(np.empty((2, 0)),
                                                names=['t_stamp', 'label'])
        empty_trajs = pd.DataFrame(np.empty((0, len(columns))),
                                   index=empty_index,
                                   columns=columns)
        return cls(empty_trajs)

    def check_trajs_df_structure(self, index=None, columns=None):
        """Check wether trajectories contains a specified structure.

        Parameters
        ----------
        index : list
            Index names (order is important)
        columns : list
            Column names (order does not matter here)

        Raises
        ------
        ValueError in both case
        """

        error_mess = "Trajectories does not contain correct indexes : {}"
        if index and self.index.names != index:
            raise ValueError(error_mess.format(index))

        error_mess = "Trajectories does not contain correct columns : {}"
        if columns:
            columns = set(columns)
            if not columns.issubset(set(self.columns)):
                raise ValueError(error_mess.format(columns))

    # Trajs getter methods

    @property
    def t_stamps(self):
        return self.index.get_level_values('t_stamp').unique()

    @property
    def labels(self):
        return self.index.get_level_values('label').unique()

    @property
    def segment_idxs(self):
        return self.groupby(level='label').groups

    @property
    def iter_segments(self):
        for lbl, idxs in self.segment_idxs.items():
            yield lbl, self.loc[idxs]

    def get_segments(self):
        """A segment contains all the data from `self.trajs` with

        Returns
        -------
        A dict with labels as keys and segments as values
        """
        return {key: segment for key, segment
                in self.iter_segments}

    def get_longest_segments(self, n):
        """Get the n th longest segments label indexes.

        Parameters
        ----------
        n : int
        """
        idxs = self.segment_idxs
        return list(dict(sorted(idxs.items(), key=lambda x: len(x[1]))[-n:]).keys())

    def get_shortest_segments(self, n):
        """Get the n th shortest segments label indexes.

        Parameters
        ----------
        n : int
        """
        idxs = self.segment_idxs
        return list(dict(sorted(idxs.items(), key=lambda x: len(x[1]))[:n]).keys())

    def copy(self):
        """
        """

        trajs = super(self.__class__, self).copy()
        return Trajectories(trajs)

    def get_colors(self, cmap="hsv", alpha=None, rgba=False):
        '''Get color for each label.

        Parameters
        ----------
        cmap : string
            See http://matplotlib.org/examples/color/colormaps_reference.html for a list of
            available colormap.
        alpha : float
            Between 0 and 1 to add transparency on color.
        rgba : bool
            If True return RGBA tuple for each color. If False return HTML color code.

        Returns
        -------
        dict of `label : color` pairs for each segment.
        '''
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap(name=cmap)

        n = len(self.labels)
        ite = zip(np.linspace(0, 0.9, n), self.labels)
        colors = {label: cmap(i, alpha=alpha) for i, label in ite}

        if not rgba:
            def get_hex(rgba):
                rgba = np.round(np.array(rgba) * 255).astype('int')
                if not alpha:
                    rgba = rgba[:3]
                return "#" + "".join(['{:02X}'.format(a) for a in rgba])

            colors = {label: get_hex(color) for label, color in colors.items()}

        return colors

    # Segment / spot modification methods

    def remove_segments(self, segments_idx, inplace=True):
        """Remove segments from trajectories.

        Parameters
        ----------
        segments_idx : list
            List of label to remove
        """
        return self.drop(segments_idx, level='label', inplace=inplace)

    def merge_label_safe(self, traj, id=None):
        """Merge traj to self trajectories taking care to not mix labels between them.

        Parameters
        ----------
        traj : :class:`pandas.DataFrame`
        """

        traj = traj.reset_index()
        self = self.reset_index()

        self_label = set(self['label'])
        traj_label = set(traj['label'])

        same_labels = self_label.intersection(traj_label)

        if same_labels:
            new_label_start = max(traj_label.union(self_label)) + 1
            new_labels = np.arange(new_label_start, new_label_start + len(same_labels))
            self['label'] = self['label'].replace(list(same_labels), new_labels)

        if id:
            self['id'] = id[0]
            traj['id'] = id[1]

        new_trajs = pd.concat([self, traj])

        # Relabel from zero
        old_lbls = new_trajs['label']
        nu_lbls = old_lbls.astype(np.uint16).copy()
        for n, uv in enumerate(old_lbls.unique()):
            nu_lbls[old_lbls == uv] = n

        new_trajs['label'] = nu_lbls

        new_trajs.set_index(['t_stamp', 'label'], inplace=True)
        new_trajs.sort_index(inplace=True)

        return new_trajs

    def add_spots(self):
        """
        """
        pass

    # All trajectories modification methods

    def reverse(self):
        """Reverse trajectories.

        Returns
        -------
        A copy of current :class:`sktracker.trajectories.Trajectories`
        """

        trajs = self.copy()
        trajs.reset_index(inplace=True)
        trajs['t_stamp'] = trajs['t_stamp'] * -1
        trajs['t'] = trajs['t'] * -1
        trajs.sort('t_stamp', inplace=True)
        trajs.set_index(['t_stamp', 'label'], inplace=True)
        return trajs

    def relabel(self, new_labels=None, inplace=True):
        """
        Sets the trajectory index `label` to new values.

        Parameters
        ----------
        new_labels: :class:`numpy.ndarray` or None, default None
            The new label. If it is not provided, the function
            will look for a column named "new_label" in `trajs` and use this
            as the new label index

        """
        if new_labels is not None:
            self['new_label'] = new_labels

        try:
            self.set_index('new_label', append=True, inplace=True)
        except KeyError:
            err = ('''Column "new_label" was not found in `trajs` and none'''
                   ''' was provided''')
            raise KeyError(err)

        self.reset_index(level='label', drop=True, inplace=True)
        self.index.set_names(['t_stamp', 'label'], inplace=True)
        self.sort_index(inplace=True)
        # self.sortlevel('t_stamp', inplace=True)
        self.relabel_fromzero('label', inplace=inplace)

    def relabel_fromzero(self, level, inplace=False):
        """
        Parameters
        ----------
        level : str
        inplace : bool

        Returns
        -------
        trajs
        """

        old_lbls = self.index.get_level_values(level)
        nu_lbls = old_lbls.values.astype(np.uint16).copy()
        for n, uv in enumerate(old_lbls.unique()):
            nu_lbls[old_lbls == uv] = n

        if not inplace:
            trajs = self.copy()
        else:
            trajs = self

        trajs['new_label'] = nu_lbls

        trajs.set_index('new_label', append=True, inplace=True)
        trajs.reset_index(level, drop=True, inplace=True)

        index_names = list(trajs.index.names)
        index_names[index_names.index('new_label')] = level
        trajs.index.set_names(['t_stamp', 'label'], inplace=True)

        return trajs

    def time_interpolate(self, sampling=1, s=0, k=3, time_step=None,
                         coords=['x', 'y', 'z']):
        """
        Interpolates each segment of the trajectories along time
        using `scipy.interpolate.splrep`

        Parameters
        ----------
        sampling : int,
            Must be higher or equal than 1, will add `sampling - 1` extra points
            between two consecutive original data point. Sub-sampling is not supported.
        coords : tuple of column names, default `('x', 'y', 'z')`
           the coordinates to interpolate.
         s : float
            A smoothing condition. The amount of smoothness is determined by
            satisfying the conditions: sum((w * (y - g))**2,axis=0) <= s where g(x)
            is the smoothed interpolation of (x,y). The user can use s to control
            the tradeoff between closeness and smoothness of fit. Larger s means
            more smoothing while smaller values of s indicate less smoothing.
            Recommended values of s depend on the weights, w. If the weights
            represent the inverse of the standard-deviation of y, then a good s
            value should be found in the range (m-sqrt(2*m),m+sqrt(2*m)) where m is
            the number of datapoints in x, y, and w. default : s=m-sqrt(2*m) if
            weights are supplied. s = 0.0 (interpolating) if no weights are
            supplied.
        k : int
           The order of the spline fit. It is recommended to use cubic splines.
           Even order splines should be avoided especially with small s values.
           1 <= k <= 5

        Returns
        -------
        interpolated : a :class:`Trajectories` instance
           The interpolated values, with column names given by `coords`
           plus the computed speeds (first order derivative) and accelarations
           (second order derivative) if `k` > 2

        Notes
        -----
        The return trajectories are NOT indexed like the input (in particular for `t_stamp`)

        The `s` and `k` arguments are passed to `scipy.interpolate.splrep`, see this
             function documentation for more details
        If a segment is too short to be interpolated with the passed order `k`, the order
             will be automatically diminished
        Segments with only one point will be returned as is


        """
        if sampling is None and time_step is not None:
            log.warning(''' The `time_step` argument is deprecated (too fuzzy)'''
                        '''Use the `sampling` argument instead ''')
            dts = self.get_segments()[0].t.diff().dropna()
            dt = np.unique(dts)[0]
            if time_step > dt:
                raise NotImplementedError('''Subsampling is not supported, '''
                                          '''give a time_step bigger than the original''')
            sampling = np.int(dt/time_step)
            log.warning('''sampling was set to {} ({}/{})'''
                        .format(sampling, dt, time_step))
        interpolated = Trajectories(time_interpolate_(self, sampling, s, k, coords))
        return Trajectories(interpolated)

    def scale(self, factors, coords=['x', 'y', 'z'], inplace=False):
        '''Multiplies the columns given in coords by the values given in factors.
        The `factors` and `columns` must have the same length

        Parameters
        ----------
        factors : sequence of floats
            Values by which each colum will be multiplied
        columns : sequence of column indices, default ['x', 'y', 'z']
            Name of the columns to be scaled by factors
        inplace : bool, optional, default False
            If True, modifies the trajectories inplace, else returns a copy

        Returns
        -------

        The original trajectories scaled or a copy
        '''

        if len(factors) != len(coords):
            raise ValueError('''Arguments factors and coords must be of same length''')
        trajs = self if inplace else self.copy()
        for factor, coord in zip(factors, coords):
            trajs[coord] = trajs[coord] * factor
        return trajs

    def project(self, ref_idx,
                coords=['x', 'y'],
                keep_first_time=False,
                reference=None,
                inplace=False,
                progress=False):
        """Project each point on a line specified by two points.

        Parameters
        ----------
        ref_idx : list of int (length should be 2)
            This two series of points will be used as a reference line to make projection.
        coords :
            Column names.
        keep_first_time : bool
            By default reference line is computed for each timepoint. If you want to keep the first
            time stamp as reference line, put this parameter to True.
        reference :
            TODO
        inplace : bool
            Add projection inplace or to a new Trajectories
        progress : bool
            Show progress bar.

        Returns
        -------
        Trajectories with two new columns : 'x_proj', and 'y_proj'.
        """

        trajs = self if inplace else self.copy()
        trajs.sort_index(inplace=True)

        # First we check if both ref_idx are present in ALL t_stamp
        n_t = trajs.index.get_level_values('t_stamp').unique().shape[0]

        if len(coords) not in (2, 3):
            mess = "Length of coords {} is {}. Not supported number of dimensions"
            raise ValueError(mess.format(coords, len(coords)))

        trajs['x_proj'] = np.nan
        trajs['y_proj'] = np.nan

        ite = trajs.swaplevel("label", "t_stamp").groupby(level='t_stamp')
        A = None
        first_time = True
        for i, (t_stamp, peaks) in enumerate(ite):

            if progress:
                print_progress(i * 100 / n_t)

            p1 = peaks.loc[ref_idx[0]][coords]
            p2 = peaks.loc[ref_idx[1]][coords]

            if p1.empty or p2.empty:
                trajs.loc[t_stamp, 'x_proj'] = np.nan
                trajs.loc[t_stamp, 'y_proj'] = np.nan
            else:
                if not keep_first_time or (keep_first_time and first_time):

                    if reference is None:
                        ref = (p1 + p2) / 2
                        vec = (p1 - ref).values[0]
                    else:
                        ref = [p1, p2][reference]
                        vec = (((p1 + p2) / 2) - ref).values[0]

                    ref = ref.values[0]
                    A = transformations_matrix(ref, vec)
                    first_time = False

                # Add an extra column if coords has two dimensions
                if len(coords) == 2:
                    peaks_values = np.zeros((peaks[coords].shape[0],
                                            peaks[coords].shape[1] + 1)) + 1
                    peaks_values[:, :-1] = peaks[coords].values
                elif len(coords) == 3:
                    peaks_values = peaks[coords].values

                # Apply the transformation matrix
                peaks_values = np.dot(peaks_values, A)[:, :-1]

                trajs.loc[t_stamp, 'x_proj'] = peaks_values[:, 0]
                trajs.loc[t_stamp, 'y_proj'] = peaks_values[:, 1]

        if progress:
            print_progress(-1)

        if np.abs(trajs.x_proj).mean() < np.abs(trajs.y_proj).mean():
            trajs.loc[:, ['x_proj', 'y_proj']] = trajs.loc[:, ['y_proj', 'x_proj']].values

        if not inplace:
            return trajs

    # Measures

    def get_mean_distances(self, group_args={'by': 'true_label'},
                           coords=['x', 'y', 'z']):
        """Return the mean distances between each timepoints. Objects are grouped
        following group_args parameters.

        Parameters
        ----------
        group_args : dict
            Used to group objects with :meth:`pandas.DataFrame.groupby`.
        coords : list
            Column names used to compute euclidean distance.

        Returns
        -------
        mean_dist : :class:`pandas.DataFrame`
        """

        def get_euclidean_distance(vec):
            vec = vec.loc[:, coords].values
            dist = (vec[:-1] - vec[1:]) ** 2
            dist = dist.sum(axis=-1)
            dist = np.sqrt(dist)
            return pd.DataFrame(dist, columns=['distance'])

        groups = self.groupby(**group_args)
        distances = groups.apply(get_euclidean_distance)
        mean_dist = distances.groupby(level=0).mean()

        return mean_dist

    def all_speeds(self, coords=['x', 'y', 'z']):
        """Get all speeds in trajectories between each t_stamp.
        """
        t_stamp = self.index.get_level_values('t_stamp').unique()
        speeds = []

        for t1, t2 in zip(t_stamp[:-1], t_stamp[1:]):
            p1 = self.loc[t1]
            p2 = self.loc[t2]
            dt = p2['t'].unique()[0] - p1['t'].unique()[0]

            d = sp.spatial.distance.cdist(p1.loc[:, coords], p2.loc[:, coords]).flatten()
            speeds += (d / dt).tolist()

        return np.array(speeds)

    # Visualization methods

    def show(self, xaxis='t',
             yaxis='x',
             groupby_args={'level': "label"},
             ax=None, **kwargs):  # pragma: no cover
        """Show trajectories

        Parameters
        ----------
        xaxis : str
        yaxis : str
        groupby : dict
            How to group trajectories
        ax : :class:`matplotlib.axes.Axes`
            None will create a new one.
        **kwargs are passed to the plot function

        Returns
        -------
        :class:`matplotlib.axes.Axes`

        Examples
        --------
        >>> from sktracker import data
        >>> from sktracker.tracker.solver import ByFrameSolver
        >>> import matplotlib.pylab as plt
        >>> true_trajs = data.brownian_trajectories_generator(p_disapear=0.1)
        >>> solver = ByFrameSolver.for_brownian_motion(true_trajs, max_speed=2)
        >>> trajs = solver.track(progress_bar=False)
        >>> fig, (ax1, ax2) = plt.subplots(nrows=2)
        >>> ax1 = trajs.show(xaxis='t', yaxis='x', groupby_args={'level': "label"}, ax=ax1)
        >>> ax2 = trajs.show(xaxis='t', yaxis='x', groupby_args={'by': "true_label"}, ax=ax2)

        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()
        colors = self.get_colors()
        gp = self.groupby(**groupby_args).groups

        # Set default kwargs if they are not provided
        # Unfortunately you can't pass somthing as '-o'
        # as a single linestyle kwarg

        if ((kwargs.get('ls') is None)
           and (kwargs.get('linestyle') is None)):
            kwargs['ls'] = '-'
        if kwargs.get('marker') is None:
            kwargs['marker'] = 'o'
        if ((kwargs.get('c') is None) and (kwargs.get('color') is None)):
            auto_color = True
        else:
            auto_color = False

        for k, v in gp.items():
            traj = self.loc[v]
            if auto_color:
                c = colors[v[0][1]]  # that's the label
                kwargs['color'] = c
            ax.plot(traj[xaxis], traj[yaxis], **kwargs)

        ax.set_xlabel(xaxis)
        ax.set_ylabel(yaxis)
        ax.set_title(str(groupby_args))

        return ax

# Register the trajectories for storing in HDFStore
# as a regular DataFrame
pytables._TYPE_MAP[Trajectories] = 'frame'
