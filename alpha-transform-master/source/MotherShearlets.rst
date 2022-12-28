Implementation of different "Mother Shearlets"
==============================================

.. automodule:: MotherShearlets

The class ``BumpFunction``
**************************
.. autoclass:: BumpFunction

The class ``MotherShearlet``
****************************
.. autoclass:: MotherShearlet

:class:`BumpFunction`-related helper functions
**********************************************
.. autofunction:: translate
.. autofunction:: scale
.. autofunction:: flip

Meyer mother shearlet implementation
************************************

The object ``MotherShearlets.MeyerMotherShearlet`` is an instance
of :class:`MotherShearlet`.

It uses

* :func:`meyer_low_pass` as the generating function for the low-pass part,
* :func:`meyer_scale_function` as the *scale-sensitive* generating function.
* :func:`meyer_direction_function` as the *direction sensitive* generating
  function.

.. autofunction:: meyer
.. autofunction:: meyer_scale_function
.. autofunction:: meyer_low_pass
.. autofunction:: meyer_direction_function


Haeuser mother shearlet implementation
**************************************

The *Häuser mother shearlet* is the one which is used in the paper
|FFST| by S. Häuser and G. Steidl. In our implementation, it is
represented by the object ``MotherShearlets.HaeuserMotherShearlet``,
which is an instance of :class:`MotherShearlet`.

It uses

* :func:`meyer_low_pass` as the generating function for the low-pass part,
* :func:`haeuser_scale_function` as the *scale-sensitive* generating function,
* :func:`haeuser_direction_function` as the *direction sensitive* generating
  function.

.. autofunction:: haeuser_scale_function
.. autofunction:: haeuser_direction_function


Indicator mother shearlet implementation
****************************************

The "indicator" mother shearlet is a mother shearlet which is used
purely for testing purposes. The low-pass part and the mother shearlet
determined by its generating functions are indicator functions in the
Fourier domain. Hence, a shearlet system using this mother shearlet
will have **very bad** time localization.

The "indicator" mother shearlet is represented by the object
``MotherShearlets.IndicatorMotherShearlet``, which is an instance of
:class:`MotherShearlet`. It uses

* :math:`\chi_{[-1,1]}` as the *low-pass* part of the set of generating
  functions,
* :math:`\chi_{[1,2]}` as the *scale-sensitive* generating function, and
* :math:`\chi_{[-1/2, 1/2]}` as the *direction sensitive* generating function.

.. autofunction:: indicator_scale_function
.. autofunction:: indicator_low_pass_function


.. _fast finite shearlet transform: https://arxiv.org/abs/1202.1773
.. |FFST| replace:: `Fast Finite Shearlet Transform`_
.. _cartoon approximation with α-curvelets: https://arxiv.org/abs/1404.1043/
.. |Cartoon| replace:: `Cartoon approximation with α-Curvelets`_
