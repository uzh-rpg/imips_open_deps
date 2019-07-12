# Copyright (C) 2019 Titus Cieslewski, RPG, University of Zurich, Switzerland
#   You can contact the author at <titus at ifi dot uzh dot ch>
# Copyright (C) 2019 Davide Scaramuzza, RPG, University of Zurich, Switzerland
#
# This file is part of imips_open_deps.
#
# imips_open_deps is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# imips_open_deps is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with imips_open_deps. If not, see <http:#www.gnu.org/licenses/>.

from __future__ import print_function
import numpy as np
import os
import sys
import tensorflow as tf

class StepError(Exception):
    pass

# For a more general-purpose (but more boilerplaty) neural net tracking, see TF
# summaries:
# https://www.tensorflow.org/guide/summaries_and_tensorboard
# 
# Example usage:
# def step_fun(sess):
#     im = ds.buildTriplet()
#     sess.run(g.step, feed_dict=g.fdict(im))
# 
# def val_fun(sess):
#     vlosses = []
#     for val_trip in val_trips:
#         vlosses.append(sess.run(g.loss, feed_dict=g.fdict(val_trip)))
#     loss = np.mean(np.array(vlosses))
#     return [loss]
# 
# trainer = Trainer(step_fun, val_fun, 'checkpoints', 'test', check_every=10)
# stats = trainer.trainUpToNumIts(sess, 400)
# 
# plt.plot(stats[:, 0], stats[:, 1])
# plt.show()
    

class Trainer(object):
    def __init__(self, step_fun, val_fun, outdir, filename, check_every=100,
                 step_has_prints=False, best_index=None, best_is_max=False):
        ''' step_fun and val_fun take a tf session as argument.
        step_fun can throw a StepError to abort training.
        val_fun returns a list of floats which are appended to stats. 
        val_fun and checkpoint save is called every check_every steps.
        best_index is zero-based of the list returned by val_fun. '''
        self._step_fun = step_fun
        self._val_fun = val_fun
        self._outdir = outdir
        self._filename = filename
        self._check_every = check_every
        self._step_has_prints = step_has_prints
        self._best_index = best_index
        self._best_is_max = best_is_max
        
        self._saver = tf.train.Saver()
        self._outfile = os.path.join(self._outdir, filename)
        self._statfile = '%s_stats' % self._outfile
        self._bestfile = self._outfile + '_best'
        self._beststatfile = '%s_stats' % self._bestfile
        self._initialized = False
    
    def init(self, sess):
        ''' Optional funtion to be called before training for initialization
        '''
        assert not self._initialized
        # See whether we need to resume or initialize.
        try:
            self.stats = np.loadtxt(self._statfile)
            if len(self.stats.shape) == 1:
                self.stats = np.expand_dims(self.stats, 0)
            self.step_base = self.stats[-1, 0]
            assert tf.train.checkpoint_exists(self._outfile)
            self._saver.restore(sess, self._outfile)
            print('[Trainer] Resuming from %s...' % self._outfile)
        except IOError:
            self.stats = None
            self.step_base = 0       
            sess.run(tf.global_variables_initializer())
            print('[Trainer] Random initialization.')
        self._initialized = True
    
    def trainUpToNumIts(self, sess, its):
        if not self._initialized:
            self.init(sess)
        
        for step in xrange(int(self.step_base), its):
            if not self._step_has_prints:
                print('\r[Trainer] Step %d, %d left:' % (step + 1, its - step - 1),
                      end='')
                sys.stdout.flush()
            else:
                print('\n[Trainer] Step %d, %d left:' % 
                      (step + 1, its - step - 1))
            try:
                self._step_fun(sess)
            except StepError:
                break
            if (step == 0) or ((step + 1) % self._check_every == 0):
                stats_i = [step + 1] + self._val_fun(sess)
                print('\n[Trainer] Validataion stats: ')
                print(stats_i)
                if self.stats is None:
                    self.stats = np.array(stats_i).reshape((1, -1))
                else:
                    self.stats = np.vstack((self.stats, stats_i))
                np.savetxt(self._statfile, self.stats)
                self._saver.save(sess, self._outfile)

                if self._best_index is not None:
                    best_index = self._best_index + 1
                    if self._best_is_max:
                        bestfun = np.max
                    else:
                        bestfun = np.min
                    if stats_i[best_index] == bestfun(
                            self.stats[:, best_index]):
                        np.savetxt(self._beststatfile, self.stats)
                        self._saver.save(sess, self._bestfile)

        return self.stats
