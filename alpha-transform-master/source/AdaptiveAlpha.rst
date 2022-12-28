Choosing α adaptively
=====================

The module ``AdaptiveAlpha.py`` can be used to choose the optimal value
of the *anisotropy parameter* α, with respect to a given optimality
*criterion* and a given family of images.

Currently, the following *criteria* are implemented:

#. the |AAR| (AAR),
#. the |MAE| (MAE),
#. the |TDP| (TDP).

In the following, these different criteria are described in greater detail.
Finally, we document the auxiliary functions used to implement the criteria.

Documentation of the optimality criteria
****************************************

.. automodule:: AdaptiveAlpha
   :members: optimize_AAR, optimize_MAE, optimize_denoising

Documentation of auxiliary functions
************************************

.. automodule:: AdaptiveAlpha
   :members:
   :exclude-members: optimize_AAR, optimize_MAE, optimize_denoising



.. |AAR| replace:: :func:`asymptotic approximation rate <optimize_AAR>`
.. |MAE| replace:: :func:`mean approximation error <optimize_MAE>`
.. |TDP| replace:: :func:`thresholding denoising performance <optimize_denoising>`
