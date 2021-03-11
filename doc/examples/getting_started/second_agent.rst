**Steps:** [ :doc:`install <install>` | :doc:`jax <prereq_jax>` | :doc:`haiku <prereq_haiku>` | :doc:`q-learning <first_agent>` | *dqn* | :doc:`ppo <third_agent>` | :doc:`next_steps <next_steps>` ]

DQN on CartPole
===============


In this example we build a slightly more sophisticated agent known as
:doc:`DQN </examples/stubs/dqn>` (`paper <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_).


.. image:: /_static/img/cartpole.gif
    :alt: CartPole environment solved.
    :align: center


You'll solve the **CartPole** environment, in which the agent is tasked to balance an pole fixed to
a cart. The way the agent does this is by nudging the cart either to the left or the right at every
time step.

This example is nice, because it shows a few more components. Most notably, it introduces the notion
of a *target network* and the use of an *experience-replay buffer*.

Just as in the first example, you may either hit the Google Colab button or download and run the
script on your local machine.

----

:download:`dqn.py </examples/cartpole/dqn.py>`

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Google Colab
    :target: https://colab.research.google.com/github/coax-dev/coax/blob/main/doc/_notebooks/cartpole/dqn.ipynb

.. literalinclude:: /examples/cartpole/dqn.py
