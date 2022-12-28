#!/usr/bin/env python3

import itertools
import numpy as np
# the next two lines are only for working remotely (using ssh)
# import matplotlib
# matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import gc
from AlphaTransform import AlphaShearletTransform as AST
from PIL import Image
import math
from AlphaTransformUtility import (threshold,
                                   image_load,
                                   denoise)
# NOTE: If one wants to disable tqdm, comment the following line
from tqdm import tqdm
# and uncomment the following function
# def tqdm(iterable, *var, **kw):
#     return iterable


def flexible_linear_regression(xs, ys):
    r"""
    Given sequences ``xs`` and  ``ys`` of values :math:`[x_0,\dots, x_{n-1}]`
    and :math:`[y_0, \dots, y_{n-1}]`, this function computes the best
    (affine)-linear approximation to the sequence
    :math:`[(x_0,y_0),\dots,(x_{n-1},y_{n-1})]` and
    returns the y-intercept and slope of this best linear approximation.

    **Required parameters**

    :param list xs:
        List of real values :math:`[x_0, \dots, x_{n-1}]` which determine
        the x-values of the sequence of points on which the linear regression
        should be performed.

    :param list ys:
        List of real values :math:`[y_0, \dots, y_{n-1}]` which determine
        the y-values of the sequence of points on which the linear regression
        should be performed.

        .. note::
            The lists ``xs`` and ``ys`` are required to have the same length,
            which should at least be 2.

    **Return values**

    :return:
        A tuple ``(y_intercept, slope)``, where

        * ``y_intercept`` is the y-intercept (a :class:`float`) of the
          best (affine)-linear approximation to the given points.

        * ``slope`` is the slope (a :class:`float`) of the best
          (affine)-linear approximation to the given points.

        In total, the best (affine)-linear approximation to the given points
        is given by :math:`y = \mathrm{y\_intercept} + \mathrm{slope} \cdot x`.
    """
    n = len(ys)
    assert n == len(xs)
    assert n >= 2, ("Can't do linear regression on "
                    "a sequence with at most 1 element.")
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    covariance = np.dot(xs, ys) / n - x_mean * y_mean
    x_variance = np.dot(xs, xs) / n - x_mean**2
    slope = covariance / x_variance
    abscissa = y_mean - slope * x_mean
    return (abscissa, slope)


def linear_regression(ys):
    r"""
    Given a sequence ``ys`` of values :math:`[y_0, \dots, y_{n-1}]`, which are
    interpreted as the :math:`y`-coordinates of
    :math:`[(0, y_1), \dots, (n-1, y_{n-1})]`, this function
    computes the best (affine)-linear approximation and returns the y-intercept
    and slope of the best (affine)-linear approximation, as well as the
    sequence of errors.

    .. note::
        This function is essentially a more optimized version of the call
        ``flexible_linear_regression(range(len(ys)), ys)``. The main difference
        is that ``linear_regression`` also returns the sequence of errors.

    **Required Parameters**

    :param list ys:
        List of real values interpreted as :math:`[y_0, \dots, y_{n-1}]` for
        the linear regression on :math:`[(0, y_0), \dots, (n-1, y_{n-1})]`,
        where :math:`n=` ``len(ys)``.

    **Return values**

    :return:
        A tuple ``(errors, y_intercept, slope)``, where

        * ``errors`` is a :class:`list` of the differences between the given
          values (``ys``) and the values of the linear regression at
          :math:`x`-values :math:`0, \hdots, n-1`
          with :math:`n =` ``len(ys)``.

        * ``y_intercept`` is the y-intercept (a :class:`float`) of the
          best (affine)-linear approximation.

        * ``slope`` is the slope (a :class:`float`) of the best
          (affine)-linear approximation.

        In total, the best (affine)-linear approximation to the sequence of
        points :math:`[(0, y_0), \dots, (n-1, y_{n-1})]` is given by
        :math:`y = \mathrm{y\_intercept} + \mathrm{slope} \cdot x`.
    """
    n = len(ys)
    assert n >= 2, ("Can't do linear regression on "
                    "a sequence with at most 1 element.")
    xs = np.arange(n) - (n - 1) / 2  # the mean of np.arange(n) is (n-1)/2
    zs = np.array(ys, dtype=float)
    mean = np.mean(zs)
    zs -= mean
    slope = np.sum(xs * zs) / np.sum(xs ** 2)
    errors = zs - slope * xs
    abscissa = mean - slope * (n - 1) / 2
    return (errors, abscissa, slope)


def common_linear_segmentation(list_of_ys, max_error, direction):
    r"""
    This function performs a *sliding window piecewise linear
    time series segmentation*, based on a certain
    `Stackoverflow post <http://www.stackoverflow.com/questions/24872314>`_,
    simultaneously over a given set of time series.

    The parameter ``list_of_ys`` contains the set of time series, i.e., it is
    of the form :math:`\mathrm{list\_of\_ys} =
    [\mathrm{ys}^{(0)}, \dots, \mathrm{ys}^{(N)}]`,
    where each element :math:`\mathrm{ys}^{(i)}` is itself a list with real
    values :math:`\mathrm{ys}^{(i)} = [y_0^{(i)}, \dots, y_{n-1}^{(i)}]`,
    representing a time series. Note that the length ``n`` of the time series
    should be independent of ``i``.
    The function then computes a sequence :math:`(x_1, ..., x_\ell)` of break
    points such that between each pair of consecutive break points, each of the
    sequences :math:`\mathrm{ys}^{(i)}` is "almost linear".

    **Required parameters**

    :param list list_of_ys:
            List of time series (of common length) which will be simultaneously
            split into "almost linear" parts using a common segmentation.
            For more details on the form of ``list_of_ys``, see the description
            from above.

    :param float max_error:
            A positive real number. The interval :math:`[0, \dots, n]`,
            where `n` is the *common* length of all time series, is split into
            a number of intervals (segments) such that on each segment and for
            each of the time series ``ys``, the best (affine)-linear
            approximation (in the sense of linear regression) to the time
            series ``ys`` itself has a uniform error of at most
            ``max_error * (max(ys) - min(ys))``.

            Hence, for a given value of ``max_error``, the allowed tolerance
            is still weighted with the actual *spread* of the values in each
            of the time series. This ensures that a segmentation of a scaled
            version of the given time series will result in the same
            segmentation as for the original time series.

    :param string direction:
            Either ``'forward'`` or ``'reversed'``.

            For ordinary *sliding window piecewise linear segmentation*,
            one starts at the beginning of the time series (i.e., at 0) and
            enlarges the current segment until the criterion described above
            (see ``max_error``) fails; then one starts the next segment.
            If ``direction`` is ``'forward'``, this is exactly what this
            function does. If ``direction`` is ``'reversed'``, then we start
            at the **end** of the time series instead of at the beginning.

    **Return values**

    :return:
        A tuple ``(breaks, slopes)``, where

        * ``breaks`` is a :class:`list` of nonnegative integers which is of
          the form :math:`[x_0=0, x_1, ..., x_{k},x_{\ell+1}=n]`. This list
          encode the different "common almost linear segments" found by the
          function. Precisely, the i-th segment is given by
          :math:`[x_i, x_{i+1})`.

          .. note:: This interval is open on the right.

        * ``slopes`` is a :class:`list` of reals, where ``slopes[i]`` is the
          slope of the best (affine)-linear approximation to the values ``ys``
          on the i-th segment.
    """
    if direction.startswith('re'):
        list_of_ys = [ys[::-1] for ys in list_of_ys]

    common_length = len(list_of_ys[0])

    breakpoints = []

    left = 0
    while left < common_length:
        end_of_segments = []
        for ys in list_of_ys:
            value_range = np.max(ys) - np.min(ys)
            for right in range(left + 1, common_length):
                # print("left", left)
                # print("right", right)
                errors = linear_regression(ys[left:right + 1])[0]
                if np.max(np.abs(errors)) > value_range * max_error:
                    end_of_segments.append(right)
                    break
            else:  # if the loop runs to completion without 'breaking'
                end_of_segments.append(common_length)
        breakpoints.append(left)
        left = min(end_of_segments)
    breakpoints.append(common_length)

    if direction.startswith('re'):
        breakpoints = [common_length - breakpoint
                       for breakpoint in reversed(breakpoints)]

    return breakpoints

DEFAULT_COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow']


def log_plot_linear_segmentations(xs,
                                  ys_list,
                                  max_error,
                                  labels,
                                  direction='reversed',
                                  colors=None):
    r"""
    Given a set of time series (in ``ys_list``, with corresponding time stamps
    given by ``xs``), this function plots these time series in a log-log plot.
    In addition to the time series itself, the function also computes a
    segmentation on the time axis such that on each segment, each of the given
    time series is almost linear (*in the log-log plot*). On each of these
    segments, also the best linear approximation (in the log-log plot) to each
    of the time series is plotted.

    **Parameters**

    :param list ys_list:
        A list of time series with *positive* values, i.e., each element of
        ``ys_list`` should be a list of positive real numbers and all elements
        of ``ys_list`` should have the same common length.

    :param list xs:
        A list of *positive* x-values ("time stamps") associated to the time
        series in ``ys_list``. In other words, for each time series ``ys`` in
        ``ys_list``, ``xs[i]`` should be the x-value corresponding to the
        y-value ``ys[i]``.
        In particular, ``xs`` should have the same length as every element
        of ``ys_list``.

        .. warning::
            It is implicitly assumed that ``xs`` is of the form
            ``xs = [x_0 * b**i for i in range(len(xs))]``, for some ``x_0 >0``
            and some ``0 < b != 1``, since the *logarithmic* time series are
            segmented using :func:`common_linear_segmentation`. This only
            yields the same result as a piecewise linear time series
            segmentation of the time series *in a log-log plot* if the x-values
            in ``xs`` behave linearly in the log-log plot. This is equivalent
            to the stated form of ``xs``.

    :param float max_error:
        A *positive* number which determines the tolerance for splitting the
        x-axis into several segments on each of which all of the time series
        should be "almost linear" (*in a log-log plot*).

        Precisely, the segmentation is determined by calling
        ``common_linear_segmentation(log_ys_list, max_error, direction)``,
        where ``log_ys_list = [np.log10(ys) for ys in ys_list]``.

    :param list labels:
        This list (of strings) determines the label used in the plot for each
        of the time series in ``ys_list``. In particular, the length of
        ``labels`` should be the same as that of ``ys_list``.

    :param string direction:
        Either ``'reversed'`` or ``'forward'``. This parameter determines
        whether the piecewise linear time series segmentation should be started
        at the start of the time series (in case of ``direction = 'forward'``)
        or at the end of the time series (for ``direction = 'reversed'``);
        see also :func:`common_linear_segmentation`.

        Since this function is mainly to be used for plotting approximation
        rate curves and since these tend to be "more linear" at the end and
        since the important part for the **asymptotic** approximation rate is
        at the end of the time series, ``'reversed'`` is the default value.

    :param list colors:
        A list of string which determines the color to be used for plotting
        each of the time series. Once the list of colors is exhausted, it is
        traversed again at the beginning.

        If the default value (``None``) is used, a default list of colors is
        used.

    **Return value**

    :return:
        Nothing. Note that the resulting plot is not immediately shown, so that
        changes to the plot can be made. Call ``matplotlib.pyplot.show()`` to
        display the plot.
    """

    if colors is None:
        colors = DEFAULT_COLORS

    # determine common break points
    log_ys_list = [np.log10(ys) for ys in ys_list]
    common_breaks = common_linear_segmentation(log_ys_list,
                                               max_error,
                                               direction)

    for color, label, ys in zip(itertools.cycle(colors), labels, ys_list):
        _log_plot_linear_segmentation(xs,
                                      ys,
                                      common_breaks,
                                      color,
                                      label)
    plt.gca().invert_xaxis()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='best', prop={'size': 20})


def _log_plot_linear_segmentation(xs,
                                  ys,
                                  common_breaks,
                                  col=None,
                                  lab=None):
    # xs = x_base ** (- np.arange(len(ys)))
    log_xs = np.log10(xs)
    log_ys = np.log10(ys)
    # breakpoints = linear_segmentation(log_ys, max_error, direction)[0]
    for i, (left, right) in enumerate(zip(common_breaks, common_breaks[1:])):
        if right - left >= 2:
            # abscissa, slope = linear_regression(log_ys[left:right])[1:]
            abscissa, slope = flexible_linear_regression(log_xs[left:right],
                                                         log_ys[left:right])
        else:
            abscissa, slope = ys[left], 1
        xs_local = xs[left:right]
        linear_ys = 10**(slope * np.log10(xs_local) + abscissa)
        if i == 0:
            plt.plot(xs_local, linear_ys, '--', linewidth=2, color=col)
        else:
            plt.plot(xs_local, linear_ys, '--', linewidth=2, color=col)
    plt.plot(xs, ys, 'o', color=col, label=lab)


def _number_of_common_segments(list_of_ys, max_error, direction):
    return len(common_linear_segmentation(list_of_ys, max_error, direction))


def _print_alpha_list(alphas, values, offset=None, mark_min=True):
    if offset is None:
        row = "alpha = {0:.2f} : {1:+f}"
    else:
        row = "alpha = {0:.2f} : {1:+f} {2} {3:f}"

    if mark_min:
        min_i = np.argmin(values)
        marks = [' * ' if i == min_i else '   ' for i, _ in enumerate(values)]
    else:
        marks = ['' for s in values]

    for alpha, mark, value in zip(alphas, marks, values):
        if offset is None:
            print(mark + row.format(alpha, value))
        else:
            if value >= 0:
                print(mark + row.format(alpha, offset, '+', abs(value)))
            else:
                print(mark + row.format(alpha, offset, '-', abs(value)))


def optimize_asymptotic_approx_rate(error_sequences,
                                    alphas,
                                    mode='longest',
                                    max_number_of_segments=4,
                                    direction='reverse'):
    r"""
    **Required parameters**

    :param list error_sequences:
        A list of the error-sequences to compare.
        All elements of *error_sequences* must have
        the same length.

    :param list alphas:
        The respective values of alpha belonging
        to the *error_sequences*

    **Keyword parameters**

    :param int max_number_of_segments:
        An integer >= 2 which determines in how many
        linear parts each error sequence is split.

    :param string mode:
        Either 'longest' or 'last'.In case of 'longest', the function
        looks for the longest(!) 'almost linear' part in each of the
        error sequences and compares the respective slopes.
        In case of 'last', the function looks for the last(!)
        'almost linear' part (of length >= 1/(2 * max_number_of_segments)
        * length of the whole sequence) in each of the error sequences
        and compares the respective slopes.

    :param string direction:
        Either 'reverse' or 'forward'.
        This parameter influences whether the linear segmentation
        starts at the beginning or at the end of the sequence.

        For things like approximation errors, the behaviour is
        usually "more linear" at the end than at the beginning, so
        that in order to pick up this behaviour, it is mostly better
        to start at the end. Hence, 'reverse' is the default value.

    **Return values**

    :return:
        A tuple ``(i, epsilon)``, where

        * ``i`` is the index (an :class:`int`) of the best error sequence

        * ``epsilon`` is the maximal precision (a :class:`float`) which leads
          to a number of segments which is <= max_number_of_segments.
    """
    epsilon = 0.001
    epsilon_inc = 0.001
    assert max_number_of_segments >= 2

    cur_num_segments = _number_of_common_segments(error_sequences,
                                                  epsilon,
                                                  direction)
    while cur_num_segments > max_number_of_segments:
        epsilon += epsilon_inc
        cur_num_segments = _number_of_common_segments(error_sequences,
                                                      epsilon,
                                                      direction)

    # DEBUG starts here
    # print()
    # print("epsilon = {0:.3f}".format(epsilon))
    # break_slopes = [linear_segmentation(errors, epsilon, direction)
    #                 for errors in error_sequences]
    # breakpoints = [break_slope[0] for break_slope in break_slopes]
    # print("breakpoints:")
    # print_listlist(breakpoints)
    # slopes = [break_slope[1] for break_slope in break_slopes]
    # DEBUG ends here

    common_breaks = common_linear_segmentation(error_sequences,
                                               epsilon,
                                               direction)
    print("Common breakpoints:", common_breaks)
    segment_lengths = np.diff(common_breaks)

    if mode == 'longest':
        # low, up = longest_common_piece(breakpoints)
        longest_index = np.argmax(segment_lengths)
        low = common_breaks[longest_index]
        up = common_breaks[longest_index + 1]

        print("longest common segment: [", low, ",", up, ")")
        if up - low <= 1:
            # this should essentially never happen
            print("WARNING: The longest common linear part has length <= 1.")
            long_slopes = [0 for errors in error_sequences]
        else:
            long_slopes = [linear_regression(errors[low:up])[2]
                           for errors in error_sequences]
        print("long slopes:")
        _print_alpha_list(alphas, long_slopes)
        print("slope spread   :", np.max(long_slopes) - np.min(long_slopes))
        print("slope mean     :", np.mean(long_slopes))
        print("slope std. dev.:", np.std(long_slopes))
        return (np.argmin(long_slopes), epsilon)
    else:  # mode = 'last'
        # find the (index of) the last segment
        # of length >= total_length / (2* max_number_of_segments)
        # on which each of the error sequences is 'sufficiently linear'
        min_length = len(error_sequences[0]) / (2 * max_number_of_segments)
        long_segment_indices = [i for i, length in enumerate(segment_lengths)
                                if length >= min_length]
        if not long_segment_indices:
            # there is no sufficiently long common linear part
            # -> use the last linear part
            print("There is no sufficiently long common linear part.")
            low = common_breaks[-2]
            up = common_breaks[-1]
            warning_str = "Just taking the last linear part, i.e., [{0}, {1})."
            print(warning_str.format(low, up))
        else:
            last_long_index = np.max(long_segment_indices)
            low = common_breaks[last_long_index]
            up = common_breaks[last_long_index + 1]
            msg = "Last common linear part: [{0}, {1})"
            print(msg.format(low, up))
        if up - low <= 1:
            # this should essentially never happen
            print("WARNING: The last longest linear part has length <= 1.")
            last_slopes = [0 for errors in error_sequences]
        else:
            last_slopes = [linear_regression(errors[low:up])[2]
                           for errors in error_sequences]

        print()
        print("last slopes:")
        mean = np.mean(last_slopes)
        _print_alpha_list(alphas, [s - mean for s in last_slopes], offset=mean)
        # print()
        # print("slope spread   :", np.max(last_slopes) - np.min(last_slopes))
        # print("slope mean     :", np.mean(last_slopes))
        # print("slope std. dev.:", np.std(last_slopes))
        return (np.argmin(last_slopes), epsilon)


def optimize_AAR(image_paths,
                 num_scales,
                 alpha_res,
                 threshold_mode='hard',
                 num_x_values=50,
                 base=1.25,
                 show_plot=True,
                 shearlet_args=None):
    r"""
    Given a set of images :math:`f=\{f_1,...,f_N\}`, this function uses a grid
    search to determine the optimal value of the parameter alpha of an
    alpha-shearlet system by comparing the **asymptotic approximation rates**
    obtained with different alpha-shearlet systems for the given set of images.

    Precisely, the asymptotic approximation rate for a set of images
    :math:`f=\{f_1,...,f_N\}` is calculated as follows:

    #. A sequence of threshhold coefficients :math:`c=(c_j)_{j=0,...,J}` of the
       form :math:`c_j=c_0\cdot b^{-j}` is determined, where :math:`c_0>0`,
       :math:`b>1`.

    #. For each of the input images  :math:`f_i`, each alpha and each
       of the threshold parameters :math:`c_j`, the **approximation error**
       :math:`E_\alpha(f_i;c_j)
       =\|f_i-S_\alpha^{-1}\Lambda_{c_j}S_\alpha f_i\|_{L^2}`
       is calculated. Here, :math:`\Lambda_c` is a
       thresholding operator with cut-off (or threshold) :math:`c`,
       :math:`S_\alpha` is the alpha-shearlet transform and
       :math:`S_\alpha^{-1}`
       the (pseudo)-inverse of the alpha-shearlet transform.
       All images are normalized to satisfy :math:`\|f_i\|_{L^2}=1`.

    #. The mean of the approximation errors with respect to the image set
       is taken:
       :math:`E_\alpha(f;c_j)=\frac{1}{N}\sum_{i=1}^N
       \|f_i-S_\alpha^{-1}\Lambda_{c_j}S_\alpha f_i\|_{L^2}`.

    #. For each value of alpha, the **logarithm** of  :math:`E_\alpha(f;c_j)`,
       as a function of :math:`j`, is considered as a time series which is
       partitioned into **almost linear parts** using
       *piecewise linear times series segmentation*; see
       :func:`common_linear_segmentation` for more details and the techincal
       report for motivation.

    #. The value of alpha yielding the *smallest slope* (i.e., the fastest
       error decay) in the last of these almost linear parts is considered
       as the optimum.

    Many parameters (number of different alpha values, number of threshold
    parameters, etc.) of the procedure described above can be customized
    using the parameters of :func:`optimize_AAR`. These parameters are
    described in the following list.

    **Required parameters**

    :param list image_paths:
        This parameter determines the set of images to be considered.
        Precisely, ``image_paths`` should be a list of strings, where
        each string is the path of an image, i.e., of a ``.png`` or a
        ``.npy`` file.

        All of these images/numpy arrays have to be two-dimensional and
        all of the same dimension. Furthermore, ``image_paths`` should
        contain at least one path.

    :param int num_scales:
        Number of scales which the different alpha-shearlet systems should use.

    :param float alpha_res:
        This parameter determines the resolution (or density) which is used by
        the grid search to determine the optimal value of alpha.
        The different alpha values are taken uniformly over the interval [0,1],
        with sampling density ``alpha_res``.

        .. note::
            If one wants to determine the *number* of different alpha values,
            this can be done by passing ``alpha_res = 1 / (num_alphas - 1)``,
            where ``num_alphas`` is the desired number of different alpha
            values.

    **Keyword parameters**

    :param string threshold_mode:
        Either ``'hard'`` or ``'soft'``. This parameter determines whether the
        hard thresholding operator

        .. math::
            \Lambda_cx
            = \begin{cases}
                 x, & \text{if }|x|\geq c, \\
                 0, & \text{if }|x|<c,
              \end{cases}

        or the soft thresholding operator

        .. math::
            \Lambda_cx
            =\begin{cases}
                 x\cdot \frac{|x|-c}{|x|}, & \text{if }|x|\geq c, \\
                 0,                        & \text{if }|x|<c
             \end{cases}

        is used for thresholding the alpha-shearlet coefficients.

    :param int num_x_values:
        Number of different threshold parameters that are used.
        Precisely, the considered thresholds are :math:`(c_j)_{j=0,...,J}`,
        with :math:`c_j = c_0 \cdot b^{-j}`, where
        :math:`J = \mathrm{num\_x\_values} - 1` and where the base ``b`` is
        determined by the parameter ``base``. Finally,
        :math:`c_0` is choosen as the maximum value of
        :math:`\|S_\alpha f_i\|_\infty` over all images and
        all values of alpha.

    :param float base:
        Value of the basis :math:`b>1` for the calculation of the threshold
        parameters :math:`c_j=c_0\cdot b^{-j}`. See ``num_x_values`` for a
        more thorough explanation.

    :param bool show_plot:
        If this paramter is set to ``True``, executing :func:`optimize_AAR`
        will also display a log-log plot of :math:`E_\alpha (f;c)`, together
        with the associated partition into *almost linear parts*.

    :param dict shearlet_args:
        This argument can be used to determine the properties of the employed
        alpha-shearlet systems. A typical example of this argument is::

            {'subsampled' : False, 'real' : True, 'verbose' : False}

        where the chosen values of ``True`` or ``False`` can of course differ.

        .. note::
            The parameter ``shearlet_args`` is just passed as a set of keyword
            arguments to the constructor of the class
            :class:`AlphaShearletTransform
            <AlphaTransform.AlphaShearletTransform>`.
            See the documentation of that class for more details, in particular
            for the respective default values.

    **Return value**

    :return:
        The function returns that value of alpha (as a :class:`float`) which
        yields the **smallest asymptotic approximation rate**, i.e.,
        the **fastest asymptotic error-decay**, for the given set of  images.
   """
    # scrap_file = "AARScrapFile.txt"
    # data_file_pattern = "AAR_data{0:0>2d}.npy"
    # plot_file_pattern = "AAR_plot{0:0>2d}.png"
    if shearlet_args is None:
        shearlet_args = {}
    first_image = image_load(image_paths[0])
    width = first_image.shape[1]
    height = first_image.shape[0]
    del first_image

    num_alphas = int(1 / alpha_res) + 1
    alphas = np.linspace(1, 0, num_alphas)

    print("First step: Determine the maximum relevant value...")
    max_value = _get_max_relevant_value(width,
                                        height,
                                        num_scales,
                                        shearlet_args,
                                        alphas,
                                        image_paths,
                                        False)  # DO NOT ignore the low-pass
    print()
    print("Maximum relevant value: {0}".format(max_value))
    print()
    # xs = [base**k for k in range(num_x_values)]
    xs = max_value * base ** (- np.arange(num_x_values))

    recon_errors = [[[] for i, im in enumerate(image_paths)]
                    for alpha in alphas]

    print("Second step: Computing the approximation errors...")
    for i, alpha in tqdm(enumerate(alphas),
                         desc='alpha loop',
                         total=len(alphas)):
        # print("Processing alpha = {0:.2f}".format(alpha))
        # this is essentially taken from
        # 'AlphaTransformTest.py/exponential_recon_error'
        my_trafo = AST(width, height, [alpha] * num_scales, **shearlet_args)

        for j, path in tqdm(enumerate(image_paths),
                            desc='Image loop',
                            total=len(image_paths)):
            image = image_load(path)
            image /= np.linalg.norm(image)
            coeffs = my_trafo.transform(image, do_norm=True)

            for cutoff_val in tqdm(xs, total=len(xs), desc='Thresh. loop'):
                truncated_coeffs_gen = threshold(coeffs,
                                                 cutoff_val,
                                                 threshold_mode)
                rec = my_trafo.inverse_transform(truncated_coeffs_gen,
                                                 do_norm=True)
                recon_errors[i][j].append(np.linalg.norm(image - rec) /
                                          np.linalg.norm(image))
            # to prevent loss due to canceling the program, etc., save to file
            # with open(scrap_file, 'a') as alpha_error_file:
            #     print("File:", path, file=alpha_error_file)
            #     print("alpha:", alpha, file=alpha_error_file)
            #     print("number of scales:", num_scales, file=alpha_error_file)
            #     print("thresh. mode:", threshold_mode, file=alpha_error_file)
            #     print("thresh. values:", list(xs), file=alpha_error_file)
            #     print("reconstruction errors:",
            #           repr(np.array(recon_errors[i][j])),
            #           file=alpha_error_file)
            #     print("", file=alpha_error_file)
            del image
            del coeffs
            gc.collect()

        del my_trafo
        gc.collect()

    # np.save('AAR_presentation_x_axis.npy', np.array(xs))
    # np.save('AAR_presentation_data.npy', np.array(recon_errors))
    # data_file_name = find_free_file(data_file_pattern)
    # print("Saving data to {0}...".format(data_file_name))
    # np.save(data_file_name, recon_errors)
    # print()
    # print()

    print()
    print()
    print()
    print("Third step: Computing the approximation rates...")
    # now we have computed all necessary data -> perform the optimization
    # first, compute the mean over all images
    mean_errors = [[np.mean([alpha_errors[j][i]
                             for j in range(len(image_paths))])
                    for i, _ in enumerate(xs)]
                   for alpha_errors in recon_errors]
    log_mean_errors = [np.log10(alpha_mean_errors)
                       for alpha_mean_errors in mean_errors]
    # with open(scrap_file, 'a') as alpha_error_file:
    #     print("Collected mean errors:", file=alpha_error_file)
    #     print(repr(log_mean_errors), file=alpha_error_file)
    # determine the optimal values of alpha
    # best_longest_index = optimize_asymptotic_approx_rate(log_mean_errors,
    #                                                      alphas,
    #                                                      mode='longest')[0]
    best_last_index, epsilon = optimize_asymptotic_approx_rate(log_mean_errors,
                                                               alphas,
                                                               mode='last')
    print()
    print("Optimal value: alpha = {0:.2f}".format(alphas[best_last_index]))

    if show_plot:
        # plot_file_name = find_free_file(plot_file_pattern)
        # print("Final step: Saving plot to {0}".format(plot_file_name))
        # plot the data and save the plot to disk
        labels = [r"$\alpha = {0:.2f}$".format(alpha) for alpha in alphas]
        plt.figure(figsize=(24, 14))
        log_plot_linear_segmentations(xs, mean_errors, epsilon, labels)
        # plot_linear_segmentations(log_mean_errors, epsilon, labels)
        subsampled = shearlet_args.get('subsampled', False)
        font_size = 20
        plt.title(("Collected mean errors for files " +
                   os.path.basename(image_paths[0]) +
                   " - " +
                   os.path.basename(image_paths[-1]) +
                   ".\n Subsampled: {0}, Number of scales: {1}" +
                   ",\n Total number of images: {2}").format(subsampled,
                                                             num_scales,
                                                             len(image_paths)))
        plt.xlabel('Threshold level $c$', fontsize=font_size)
        y_label_text = (r'The mean (relative) $L^2$-error $E_\alpha (f; c) = '
                        r'\frac{1}{N} \sum_{i=1}^N \Vert f_i - S_\alpha^{-1} '
                        r'\Lambda_c S_\alpha f_i \Vert_{L^2}$')
        plt.ylabel(y_label_text, fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        # plt.savefig(plot_file_name)
        plt.show()
    # return the optimal alpha values
    # return (alphas[best_longest_index], alphas[best_last_index])
    return alphas[best_last_index]


def _get_max_relevant_value(width,
                            height,
                            num_scales,
                            shearlet_args,
                            alphas,
                            image_paths,
                            discard_low_pass):
    r"""
    """
    max_value = 0
    for i, alpha in tqdm(enumerate(alphas),
                         total=len(alphas),
                         desc='alpha loop'):
        my_trafo = AST(width, height, [alpha] * num_scales, **shearlet_args)
        for j, path in tqdm(enumerate(image_paths),
                            total=len(image_paths),
                            desc='image loop'):
            image = image_load(path)
            image /= np.linalg.norm(image)

            coeffs_gen = my_trafo.transform_generator(image, do_norm=True)
            if discard_low_pass:
                # discard the low-pass part, since it is
                # EXACTLY THE SAME for all alpha.
                next(coeffs_gen)
            max_values = [np.max(np.abs(coeff)) for coeff in coeffs_gen]
            max_value = max(max_value, np.max(max_values))
    del my_trafo
    del image
    gc.collect()
    return max_value


def optimize_MAE(image_paths,
                 num_scales,
                 alpha_res,
                 threshold_mode='hard',
                 num_x_values=50,
                 max_value=None,
                 show_plot=True,
                 shearlet_args=None):
    r"""
    Given a set of images :math:`f=\{f_1,...,f_N\}`, this function uses a grid
    search to determine the optimal value of the parameter alpha of an
    alpha-shearlet system by comparing the **mean approximation error**
    obtained with different alpha-shearlet systems for the given set of images.

    Precisely, the *mean approximation error* for a set of images
    :math:`f=\{f_1,...,f_N\}` is calculated as follows:

    #. A sequence of threshold coefficients :math:`c=(c_j)_{j=1,...,J}` is
       determined. In fact, the :math:`c_j` are chosen to be **uniformly
       distributed** in an interval :math:`[0, m]`, where the maximal value
       ``m`` is determined by the parameter ``max_value``.

    #. For each of the input images :math:`f_i`, each alpha and each
       of the threshold parameters :math:`c_j`, the approximation error
       :math:`E_\alpha(f_i;c_j)=
       \|f_i-S_\alpha^{-1}\Lambda_{c_j}S_\alpha f_i\|_{L^2}`
       is calculated. Here, :math:`\Lambda_c` is the
       (hard) thresholding operator with cut-off (or threshold) :math:`c`,
       :math:`S_\alpha` is the alpha-shearlet transform and
       :math:`S_\alpha^{-1}` the (pseudo)-inverse of the alpha-shearlet
       transform. All images are normalized so that :math:`\|f_i\|_{L^2}=1`.

    #. The mean of the approximation errors with respect to the image set
       and with respect to the threshold parameters is taken:
       :math:`\text{MAE}(f;\alpha)=\frac{1}{N}\sum_{i=1}^N \frac{1}{J}
       \sum_{j=1}^J\|f_i-S_\alpha^{-1}\Lambda_{c_j}S_\alpha f_i\|_{L^2}`.

    #. The value of alpha which yields the smallest value for
       :math:`\text{MAE}(f;\alpha)` is considered as the optimum.

    Many parameters (number of different alpha values, number of threshold
    parameters, etc.) of the procedure described above can be customized
    using the parameters of :func:`optimize_MAE`. These are described in
    the following list.

    **Required parameters**

    :param list image_paths:
        This parameter determines the set of images to be considered.
        Precisely, ``image_paths`` should be a list of strings, where
        each string is the path of an image, i.e., of a ``.png`` or a
        ``.npy`` file.

        All of these images/numpy arrays have to be two-dimensional and
        all of the same dimension. Furthermore, ``image_paths`` should
        contain at least one path.

    :param int num_scales:
        Number of scales which the different alpha-shearlet systems should
        use.

    :param float alpha_res:
        This parameter determines the resolution (or density) which is used by
        the grid search to determine the optimal value of alpha.
        The different alpha values are taken uniformly over the interval [0,1],
        with sampling density ``alpha_res``.

        .. note::
            If one wants to determine the *number* of different alpha values,
            this can be done by passing ``alpha_res = 1 / (num_alphas - 1)``,
            where ``num_alphas`` is the desired number of different alpha
            values.

    **Keyword parameters**

    :param string threshold_mode:
        Either ``'hard'`` or ``'soft'``. This parameter determines whether
        the hard thresholding operator

        .. math::
            \Lambda_cx
            =\begin{cases}
                x & \text{if }|x|\geq c\\
                0 & \text{if }|x|<c
             \end{cases}

        or the soft thresholding operator

        .. math::
            \Lambda_cx
            =\begin{cases}
                x\cdot \frac{|x|-c}{|x|} & \text{if }|x|\geq c\\
                0                        & \text{if }|x|<c
             \end{cases}

        is applied to each of the alpha-shearlet coefficients.

    :param int num_x_values:
        Number of different threshold parameters that are used. These are
        taken equally distributed from :math:`\{0,...,\mathrm{max\_value}\}`.

    :param float max_value:
        Maximum value of the threshold parameter.

        If the default value (``None``) is passed, ``max_value`` is taken as
        the largest absolute value of all alpha-shearlet coefficients
        (maximizing over all images and all considered values of alpha)
        **which do not belong to the low-pass part**. The reason for this
        choice is that if ``c`` is chosen greater than this threshold, then
        :math:`E_\alpha (f_i; c)` is *independent* of :math:`\alpha`.

    :param bool show_plot:
        If this parameter is set to ``True``, executing :func:`optimize_MAE`
        will display a plot of the *error curves* :math:`E_\alpha (f; c)`
        (jointly for all considered alpha values) as a function of ``c``.

    :param dict shearlet_args:
        This argument can be used to determine the properties of the employed
        alpha-shearlet systems. A typical example of this argument is::

            {'subsampled' : False, 'real' : True, 'verbose' : False}

        where the chosen values of ``True`` or ``False`` can of course differ.

        .. note::
            The parameter ``shearlet_args`` is just passed as a set of keyword
            arguments to the constructor of the class
            :class:`AlphaShearletTransform
            <AlphaTransform.AlphaShearletTransform>`.
            See the documentation of that class for more details, in particular
            for the respective default values.

    **Return value**

    :return:
        The function return the value of alpha (as a :class:`float`) which
        yields the smallest value of :math:`\text{MAE}(f;\alpha)` for the
        given set of images.
    """
    # scrap_file = "MAEScrapFile.txt"
    # data_file_pattern = "MAE_data{0:0>2d}.npy"
    # plot_file_pattern = "MAE_plot{0:0>2d}.png"

    if shearlet_args is None:
        shearlet_args = {}
    first_image = image_load(image_paths[0])
    width = first_image.shape[1]
    height = first_image.shape[0]
    del first_image

    num_alphas = int(1 / alpha_res) + 1
    alphas = np.linspace(1, 0, num_alphas)

    print("First step: Determine the maximum relevant value...")
    # For the SMBC data, the following seems to be a good choice:
    # max_value = 0.025
    if max_value is None:
        max_value = _get_max_relevant_value(width,
                                            height,
                                            num_scales,
                                            shearlet_args,
                                            alphas,
                                            image_paths,
                                            True)  # discard the low-pass part
    print()
    print("Maximum relevant value: {0}".format(max_value))

    recon_errors = [[[] for i, im in enumerate(image_paths)]
                    for alpha in alphas]
    xs = np.linspace(0, max_value, num_x_values)
    # xs = np.linspace(0, 1, num_x_values)

    print()
    print("Second step: Computing the approximation errors...")
    for i, alpha in tqdm(enumerate(alphas),
                         total=len(alphas),
                         desc='alpha loop'):
        my_trafo = AST(width, height, [alpha] * num_scales, **shearlet_args)
        for j, path in tqdm(enumerate(image_paths),
                            total=len(image_paths),
                            desc='image loop'):
            image = image_load(path)
            image /= np.linalg.norm(image)

            coeffs_normalized = my_trafo.transform(image, do_norm=True)

            for cutoff_val in tqdm(xs, desc='thresholding loop'):
                truncated_coeffs_gen = threshold(coeffs_normalized,
                                                 cutoff_val,
                                                 threshold_mode)
                rec = my_trafo.inverse_transform(truncated_coeffs_gen,
                                                 do_norm=True)
                recon_errors[i][j].append(np.linalg.norm(image - rec) /
                                          np.linalg.norm(image))
            # with open(scrap_file, 'a') as alpha_err_file:
            #     print("File:", path, file=alpha_err_file)
            #     print("alpha:", alpha, file=alpha_err_file)
            #     print("number of scales:", num_scales, file=alpha_err_file)
            #     print("threshold mode:", threshold_mode, file=alpha_err_file)
            #     print("thresholding values:", list(xs), file=alpha_err_file)
            #     print("reconstruction errors:",
            #           repr(np.array(recon_errors[i][j])),
            #           file=alpha_err_file)
            #     print("", file=alpha_err_file)
            del image
            del coeffs_normalized
            gc.collect()

        del my_trafo
        gc.collect()

    # print()
    # print()
    # data_file_name = find_free_file(data_file_pattern)
    # print("Saving data to {0}...".format(data_file_name))
    # np.save(data_file_name, recon_errors)
    # print()
    # print()

    print()
    print()
    print()
    print("Final step: Computing optimal value of alpha...")
    mean_errors = [np.mean(recon_error) for recon_error in recon_errors]
    # deviations = [np.std(recon_error) for recon_error in recon_errors]
    # mean_std = np.mean(deviations)
    # print("error std. dev.s:")
    # _print_alpha_list(alphas, deviations, mark_min=False)
    # print("mean error std. dev.:", mean_std)
    print("mean errors:")
    _print_alpha_list(alphas,
                      mean_errors - np.mean(mean_errors),
                      offset=np.mean(mean_errors))
    optimal_alpha = alphas[np.argmin(mean_errors)]
    print()
    print("Optimal value: alpha = {0:.2f}".format(optimal_alpha))
    # _print_alpha_list(alphas, mean_errors)
    # print("std. dev. of mean errors:", np.std(mean_errors))
    # print("difference to the mean mean error:")
    # _print_alpha_list(alphas, mean_errors - np.mean(mean_errors))

    if show_plot:
        # plot_file_name = find_free_file(plot_file_pattern)
        # print("Third step: Producing plot at {0}...".format(plot_file_name))
        # for j, path in enumerate(image_paths):
        plt.figure(figsize=(24, 14))
        font_size = 20
        title_str = ("Error curves for images '{0}' - '{1}'.\n"
                     "Subsampled: {2}, Number of scales: {3}.\n"
                     "Total number of images: {4}")
        plt.title(title_str.format(os.path.basename(image_paths[0]),
                                   os.path.basename(image_paths[-1]),
                                   shearlet_args.get('subsampled', False),
                                   num_scales,
                                   len(image_paths)))
        plt.ylabel(r'Mean (relative) error: '
                   r'$E_\alpha (f;c) = \frac{1}{N} \sum_{i=1}^N \Vert f_i '
                   r'- S_\alpha^{-1} \Lambda_c S_\alpha f_i \Vert_{L^2}$',
                   fontsize=font_size)
        plt.xlabel(r"Threshold level $c$", fontsize=font_size)
        for i, alpha in enumerate(alphas):
            plot_means = np.mean(recon_errors[i], axis=0)
            plt.plot(xs,
                     plot_means,
                     '-o',
                     label=r"$\alpha = {0:.2f}$".format(alpha))
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.legend(loc='best', prop={'size': 20})
        plt.grid(True)

        # plt.savefig(plot_file_name)
        plt.show()
        # print()
        # print()

    return optimal_alpha
    # return mean_errors


def optimize_denoising(image_paths,
                       num_scales,
                       alpha_res,
                       num_noise_lvls,
                       noise_min=0.02,
                       noise_max=0.4,
                       sample_size=None,
                       thresh_multiplier=None,
                       show_plot=True,
                       shearlet_args=None):
    r"""
    Given a set of images, this function uses a grid search to determine the
    optimal value of the parameter alpha of an alpha-shearlet system by
    comparing the **performance** of different alpha-shearlet systems **for**
    certain **denoising** experiments using the given set of images.

    Precisely, the denoising performance is measured as follows:
    For each of the input images :math:`f_i`, the following operations are
    performed:

    #. Gaussian noise :math:`\mathcal{N} = \mathcal{N}_\sigma` with standard
       deviation :math:`\sigma` is added to :math:`f_i` to obtain a
       *distorted image* :math:`\tilde{f_i} = f_i + \mathcal{N}`.

       .. note::
           Each :math:`f_i` is actually a **normalized** version of an
           input image, i.e., :math:`\|f_i\|_{L^2} = 1`. This normalization
           is used to ensure that the standard deviation :math:`\sigma` is
           comparable to the image itself.

    #. The alpha-shearlet transform :math:`S_\alpha \tilde{f_i}` of
       :math:`\tilde{f_i}` is computed.

    #. The alpha-shearlet coefficients are *thresholded*, yielding
       :math:`\Lambda_c S_\alpha \tilde{f_i}`, where :math:`\Lambda_c` is the
       hard thresholding operator with cut-off (or threshold) c.

       The precise value of the threshold c is actually noise- and
       scale-dependent. See below for a more detailed description.

    #. The thresholded coefficients are used to reconstruct a denoised version
       :math:`g_i` of :math:`\tilde{f_i}`.
       Precisely, :math:`g_i = S_\alpha^{-1} \Lambda_c S_\alpha \tilde{f_i}`,
       where :math:`S_\alpha^{-1}` is the (pseudo)-inverse of :math:`S_\alpha`.

    #. The error :math:`\mathrm{TDP}_\alpha (f_i;\sigma) = \|f_i - g_i\|_{L^2}`
       is computed to measure the suitability of the alpha-shearlet system for
       denoising the given image :math:`f_i`.

    This procedure is repeated for all images in the given set and for a number
    of different **noise levels** :math:`\lambda`, where the noise level is
    proprtional to the standard deviation :math:`\sigma` of the gaussian noise
    :math:`\mathcal{N}_\sigma`. In total, the suitability of the alpha-shearlet
    system for denoising the given set of images is measured by taking the mean
    over all given images and all considered noise levels, i.e., by

    .. math::
        \mathrm{TDP}_\alpha ((f_1, \dots, f_N))
        & := \frac{1}{N} \sum_{i=1}^N \frac{1}{|\Sigma|}
             \sum_{\sigma \in \Sigma} \mathrm{TDP}_\alpha (f_i; \sigma),

    where :math:`\Sigma` denotes the set of all considered standard deviations.

    Many parameters (number of different alpha values, number of different
    noise levels, etc.) of the procedure described above can be customized
    using the parameters of :func:`optimize_denoising`. These are described in
    the following list.

    **Required Parameters**

    :param list image_paths:
        This parameter determines the set of images to be considered.
        Precisely, ``image_paths`` should be a list of strings, where
        each string is the path of an image, i.e., of a ``.png`` or a
        ``.npy`` file.

        All of these images/numpy arrays have to be two-dimensional and
        all of the same dimension. Furthermore, ``image_paths`` should
        contain at least one path.

    :param int num_scales:
        Number of scales which the different alpha-shearlet systems should use.

    :param float alpha_res:
        This parameter determines the resolution (or density) which is used by
        the grid search to determine the optimal value of alpha.
        The different alpha values are taken uniformly over the interval [0,1],
        with sampling density ``alpha_res``.

        .. note::
            If one wants to determine the *number* of different alpha values,
            this can be done by passing ``alpha_res = 1 / (num_alphas - 1)``,
            where ``num_alphas`` is the desired number of different alpha
            values.

    :param int num_noise_lvls:
        Number of different noise levels that are used. These are taken equally
        distributed from the interval ``[noise_min, ..., noise_max]``.

    **Keyword parameters**

    :param float noise_min:
            Lower bound for the range from which the different **noise levels**
            are taken. Default value is :math:`0.02`.

            .. note::
                The **standard deviation** :math:`\sigma` of the noise and the
                **noise level** :math:`\lambda` are related by

                .. math:: \sigma=\lambda/\sqrt{N_1\cdot N_2},

                where :math:`N_1 \times N_2` is the common dimension of all
                considered images. This ensures

                .. math::

                    \mathbb{E} \|\mathcal{N}_\sigma\|_{L^2}^2
                    = N_1 \cdot N_2 \cdot \sigma^2 = \lambda^2,

                so that :math:`\| \mathcal{N}_\sigma\|_{L^2}` is typically
                about :math:`\lambda`. Since we are considering normalized
                images, :math:`\lambda` is thus a good measure for the noise
                to signal ratio.

    :param float noise_max:
        Upper limit for the range from which the different noise levels are
        taken. Default value is ``0.4``. See ``noise_min`` for more details.

    :param int sample_size:
        This parameter can be used to test whether *generalization* occurs,
        i.e., if the optimal value of alpha learned on a small subset
        (the *training set*) of the data yields the same value as for the
        whole data set.

        Precisely, if ``sample_size`` is passed a value different from the
        default (``None``), then ``optimize_denoising`` also determines the
        optimal value of alpha for a randomly chosen subset of the given
        images. This randomly chosen subset has ``sample_size`` elements,
        i.e., ``sample_size`` determines the size of the training set.

    :param list thresh_multiplier:
        This parameter determines how the threshold ``c`` for the hard
        thresholding operation is determined as a function of the noise
        level and of the scale.

        Precisely, the coefficients on scale ``i`` are thresholded with
        cutoff value ``sigma * thresh_multiplier[i+1]``. Here, the low-pass
        has ``i = -1``, while the other scales "begin counting" at ``i = 0``.
        Furthermore, ``sigma`` is the standard deviation of each entry of the
        noise.

        If the default value (``None``) is used, then thresh_multiplier is
        chosen as ``[3] * num_scales + [4]``, so that all scales but the
        highest use a cutoff of ``3 * sigma``, while the highest scale uses
        ``4 * sigma``.

        .. note::
            One can show (since we use the normalized alpha-shearlet
            coefficients, i.e., with :math:`\|\psi_i\|_{L^2} = 1`) that each
            coefficient :math:`(S_\alpha \mathcal{N}_\sigma)_i` is normally
            distributed with standard deviation :math:`\sigma`. Hence, choosing
            the threshold as a multiple of ``sigma`` is natural.

    :param bool show_plot:
        If this parameter is set to ``True``, executing
        :func:`optimize_denoising` will also display a plot of the average
        denoising error

        .. math::
            \frac{1}{N} \sum_{i=1}^N \mathrm{TDP}_\alpha (f_i ; \sigma)

        as a function of the noise level
        :math:`\lambda = \sqrt{N_1 \cdot N_2} \cdot \sigma` in one common plot
        for all values of alpha.

    :param dict shearlet_args:
        This argument can be used to determine the properties of the employed
        alpha-shearlet systems. A typical example of this argument is::

            {'subsampled' : False, 'real' : True, 'verbose' : False}

        where the chosen values of ``True`` or ``False`` can of course differ.

        .. note::
            The parameter ``shearlet_args`` is just passed as a set of keyword
            arguments to the constructor of the class
            :class:`AlphaShearletTransform
            <AlphaTransform.AlphaShearletTransform>`.
            See the documentation of that class for more details, in particular
            for the respective default values.

    **Return value**

    :return:
        If ``sample_size`` is ``None``, this function returns a single
        :class:`float`, namely the value of alpha yielding the best
        denoising performance on the given images.

        If ``sample_size`` is not ``None``, this function returns a
        tuple ``t`` of two floats, where ``t[0]`` is the value of alpha
        yielding the best denoising performance on *all* of the given images,
        while ``t[1]`` is the value of alpha yielding the best denoising
        performance on the randomly selected *training set* of size
        ``sample_size``.

    """
    if shearlet_args is None:
        shearlet_args = {}

    num_alphas = int(1 / alpha_res) + 1
    alphas = np.linspace(1, 0, num_alphas)

    noise_levels = np.linspace(noise_min, noise_max, num_noise_lvls)

    if thresh_multiplier is None:
        multipliers_list = [[3] * num_scales + [4]]
    else:
        multipliers_list = [thresh_multiplier]
    num_multiplier = len(multipliers_list)

    num_image = len(image_paths)
    first_image = image_load(image_paths[0])
    width = first_image.shape[1]
    height = first_image.shape[0]

    errors = np.zeros((num_image, num_multiplier, num_alphas, num_noise_lvls))
    reconstructions = np.zeros((num_alphas, height, width))

    for i, image in tqdm(enumerate(image_paths),
                         desc='image loop',
                         total=len(image_paths)):
        # if log_data:
        #     im = np.log(image_load(image))
        # else:
        im = image_load(image)
        im /= np.linalg.norm(im)
        # factor = np.max(np.abs(im))
        for k, multipliers in enumerate(multipliers_list):
            for a, alpha in tqdm(enumerate(alphas),
                                 desc='alpha loop',
                                 total=len(alphas)):
                my_trafo = AST(width, height, [alpha] * num_scales,
                               **shearlet_args)
                for j, noise_lvl in tqdm(enumerate(noise_levels),
                                         desc='noise loop',
                                         total=len(noise_levels)):
                    noise = np.random.normal(scale=noise_lvl,
                                             size=(height, width))
                    noise /= math.sqrt(height * width)
                    distorted_image = im + noise
                    reconstruction = denoise(distorted_image,
                                             my_trafo,
                                             noise_lvl,
                                             multipliers)
                    errors[i, k, a, j] = np.linalg.norm(im - reconstruction)

                    if i == 0 and j == num_noise_lvls - 1 and k == 0:
                        reconstructions[a] = reconstruction
                        orig_im = im
                        noisy_image = distorted_image

    # save the error array
    # np.save("./l2error.npy", errors)

    # save image & noisy image & reconstructions for largest noise level
    im = Image.fromarray(np.round(orig_im / np.max(orig_im) * 255))
    im.convert('LA').save('OriginalImage.png')
    im = Image.fromarray(np.round(noisy_image / np.max(noisy_image) * 255))
    im.convert('LA').save('NoisyImage.png')

    for a, alpha in enumerate(alphas):
        im = Image.fromarray(np.round(reconstructions[a] /
                                      np.max(reconstructions[a]) * 255))
        recon_file_name = ('ReconstructionNoiseLevel' +
                           str(noise_levels[num_noise_lvls - 1]) +
                           'alpha' +
                           str(alpha) +
                           '.png')
        im.convert('LA').save(recon_file_name)

    # mean error over all images
    meanerror_images = np.mean(errors, axis=0)
    # mean error over all images and all noise levels
    mean_errors = np.mean(meanerror_images, axis=2)
    # print for the first multiplier
    print()
    print()
    print()
    print()
    print("Averaged error over all images and all noise levels:")
    for a, alpha in enumerate(alphas):
        print(("alpha = {0:.2f}: {1:.4f}").format(alpha,
                                                  mean_errors[0, a]))
    print()
    opt_val_str = "Optimal value on whole set: alpha = {0:.2f}"
    print(opt_val_str.format(alphas[np.argmin(mean_errors[0])]))

    if show_plot:
        for k, multipliers in enumerate(multipliers_list):
            for a, alpha in enumerate(alphas):
                plt.plot(noise_levels,
                         meanerror_images[k, a],
                         '-o',
                         label=r"$\alpha=${al:.2f}".format(al=alpha))
            y_label_text = (r"$\frac{1}{N}\sum_{i=1}^N"
                            r"\mathrm{TDP}_\alpha(f_i;\sigma)$")
            plt.ylabel(y_label_text, fontsize=18)
            plt.xlabel(r"$\lambda = \sigma \cdot \sqrt{N_1 \cdot N_2}$",
                       fontsize=18)
            plt.grid()
            plt.legend(loc='best')
            plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.13)
            # meanerror_file_Name = ("errorl2mean" +
            #                        "reconstruction_multiplier_set_" +
            #                        str(k))
            # plt.savefig("./" + meanerror_file_Name + ".png")
            plt.show()

    if sample_size is not None:
        # choose random subset for training set
        indices = np.random.choice(range(num_image),
                                   replace=False,
                                   size=sample_size)
        # indices = np.random.randint(0, num_image, size=sample_size)
        # mean error over the "training set"
        meanerror_images_train = np.mean(errors[indices], axis=0)
        # mean error over all noise levels and all images in the "training set"
        mean_errors_train = np.mean(meanerror_images_train, axis=2)
        print()
        print()
        print()
        print()
        print("Averaged error over the training set and all noise levels:")
        for a, alpha in enumerate(alphas):
            print(("alpha = {0:.2f}: {1:.4f}").format(alpha,
                                                      mean_errors_train[0, a]))
        print()
        opt_val_str = "Optimal value on the training set: alpha = {0:.2f}"
        print(opt_val_str.format(alphas[np.argmin(mean_errors_train[0])]))

        # create plot for "training set"
        if show_plot:
            for k, multipliers in enumerate(multipliers_list):
                for a, alpha in enumerate(alphas):
                    plt.plot(noise_levels,
                             meanerror_images_train[k, a],
                             '-o',
                             label=r"$\alpha=${al:.2f}".format(al=alpha))
                y_axis_label = (r"$\frac{1}{N}\sum_{i=1}^N " +
                                r"\mathrm{TDP}_\alpha(f_i; \sigma)$")
                plt.ylabel(y_axis_label, fontsize=18)
                plt.xlabel(r"$\lambda = \sigma \cdot \sqrt{N_1 \cdot N_2}$",
                           fontsize=18)
                plt.grid()
                plt.legend(loc='best')
                plt.subplots_adjust(left=0.16, right=0.9, top=0.9, bottom=0.13)
                # meanerrorselection_file = ("errorl2_mean_train_set" +
                #                            "reconstruction_multiplier_set_"+
                #                            str(k))
                # plt.savefig("./" + meanerrorselection_file + ".png")
                # plt.close()
                plt.show()

    if sample_size is None:
        return alphas[np.argmin(mean_errors[0])]
    else:
        return (alphas[np.argmin(mean_errors[0])],
                alphas[np.argmin(mean_errors_train[0])])


if __name__ == "__main__":
    g_kwargs = {"subsampled": False,
                "real": True,
                "periodization": True,
                "parseval": False,
                "use_fftw": True,
                "verbose": True}
    # g_kwargs = {"subsampled": True,
    #             "real": False,
    #             "periodization": True,
    #             "parseval": False,
    #             "use_fftw": True,
    #             "verbose": True}
    # g_image_paths = ['../TechnicalReport/TestImages/Dust00.npy',
    #                  '../TechnicalReport/TestImages/Dust01.npy',
    #                  '../TechnicalReport/TestImages/Dust02.npy',
    #                  '../TechnicalReport/TestImages/Dust03.npy',
    #                  '../TechnicalReport/TestImages/Dust04.npy',
    #                  '../TechnicalReport/TestImages/Dust05.npy',
    #                  '../TechnicalReport/TestImages/Dust06.npy',
    #                  '../TechnicalReport/TestImages/Dust08.npy',
    #                  '../TechnicalReport/TestImages/Dust09.npy',
    #                  '../TechnicalReport/TestImages/Dust10.npy',
    #                  '../TechnicalReport/TestImages/Dust11.npy']
    # g_image_paths = ['./TestImages2/face_gray_0.png',
    #                  './TestImages2/building_gray_0.png',
    #                  './TestImages2/radialLines_clear_mod.png']
    # g_image_paths = ['./TestImages2/radialLines_clear_mod.png']
    # g_image_paths = ['./TestImages2/radialNew.png']
    # g_image_paths = [# './TestImages2/radialNew.png',
    #                  # './TestImages/random_squares_2_abs.npy',
    #                  './TestImages2/building_gray_0.png',
    #                  './TestImages2/face_gray_0.png']
    g_cropped_SBMC_path = 'SMBC/cropped'
    g_smbc_files = [os.path.join(g_cropped_SBMC_path, f)
                    for f in os.listdir(g_cropped_SBMC_path)]
    g_image_paths = sorted(g_smbc_files)
    # g_image_paths = np.random.choice(g_smbc_files,
    #                                  size=min(100, len(g_smbc_files)),
    #                                  replace=False)
    # g_image_paths = ['./TestImages/random_squares_2_abs.npy']
    # print("g_image_paths", g_image_paths)
    # g_image_paths = ['SMBC/cropped/062.png',
    #                  'SMBC/cropped/087.png',
    #                  'SMBC/cropped/081.png',
    #                  'SMBC/cropped/007.png',
    #                  'SMBC/cropped/034.png',
    #                  'SMBC/cropped/033.png',
    #                  'SMBC/cropped/068.png',
    #                  'SMBC/cropped/012.png',
    #                  'SMBC/cropped/022.png',
    #                  'SMBC/cropped/086.png']
    # g_image_paths = ['./TestImages2/balken.png']
    # for p in g_image_paths:
    #     for g_num_scales in range(5, 6):
    #         # print(optimize_AAR(g_image_paths,
    #         print(optimize_AAR([p],
    #                            g_num_scales, # number of scales
    #                            1/5, # alpha resolution
    #                            num_x_values=50,
    #                            base=1.2,
    #                            shearlet_args=g_kwargs))
    # for p in g_image_paths:
    #     for g_num_scales in range(5, 6):
    #         # print(optimize_MAE(g_image_paths,
    #         print(optimize_MAE([p],
    #                            g_num_scales, # number of scales
    #                            1/5, # alpha resolution
    #                            shearlet_args=g_kwargs))
    #         plt.close()
    #         print()
    #         print()
    #         print()
    g_num_scales = 5
    # print(optimize_MAE(g_image_paths,
    #                    g_num_scales, # number of scales
    #                    1/4, # alpha resolution
    #                    # max_value=0.03,
    #                    shearlet_args=g_kwargs))
    print(optimize_AAR(g_image_paths,
                       g_num_scales,  # number of scales
                       1 / 4,  # alpha resolution
                       base=1.25,
                       shearlet_args=g_kwargs))
    # for p in g_image_paths:
    #     print(optimize_BPDN([p],
    #                         4, # number of scales
    #                         1/5, # alpha resolution
    #                         2, # threshold multiplier
    #                         shearlet_args=g_kwargs))
    #     plt.close()
    #     print()
    #     print()
    #     print()
