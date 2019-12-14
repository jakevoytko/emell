EmEll
=====

A ML (em ell) package written in Python. The goal of this package is my own personal education.
I intend to learn different machine learning techniques by implementing them myself from
the ground up. This works in conjunction with my personal knowledge base at bitlog.com:
\[[Knowledge Base](https://www.bitlog.com/knowledge-base/machine-learning/)\].

This package is not optimized for anything but my own knowledge. Any patches / corrections /
suggestions are welcome!

Usage
=====

At the moment, it's only designed to be used in tests. The test suite can be exercised using
either...

```bash
# Exercise linting and tests
tox
# Just tests
pytest
```

Demo programs
-------------

**train_add**

Has two demos
The first has 1 neuron network with 2 inputs learning to add numbers.
This is designed to be the simplest possible neural network.

The second shows that a naive implementation of neural networks doesn't
always find a good approximation for a function that it could replicate.
It tries (and fails) to learn `y1 = .5 * x1 + x2 + 1`

```bash
python setup.py install
train_add
```

License
=======

MIT license. Go nuts.
