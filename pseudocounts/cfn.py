# coin flipping network for pseudo-counts
# takes the context vector (output of ResNet18) as input state

#code adapted from this
#https://github.com/samlobel/CFN/blob/main/bonus_based_exploration/intrinsic_motivation/intrinsic_rewards.py

import numpy as np

class CoinFlipMaker(object):
  """Their thang"""
  def __init__(self, output_dimensions, p_replace=1, only_zero_flips=False):
    self.p_replace = p_replace
    self.output_dimensions = output_dimensions
    self.only_zero_flips = only_zero_flips
    self.previous_output = self._draw()

  def _draw(self):
    if self.only_zero_flips:
      return np.zeros(self.output_dimensions, dtype=np.float32)
    return 2 * np.random.binomial(1, 0.5, size=self.output_dimensions) - 1

  def __call__(self):
    if self.only_zero_flips:
      return np.zeros(self.output_dimensions, dtype=np.float32)
    new_output = self._draw()
    new_output = np.where(
      np.random.rand(self.output_dimensions) < self.p_replace,
      new_output,
      self.previous_output
    )
    self.previous_output = new_output
    return new_output

  def reset(self):
    if self.only_zero_flips:
      self.previous_output = np.zeros(self.output_dimensions, dtype=np.float32)
    self.previous_output = self._draw()