"""Independent timing, missingness, statistical and neutralization checks."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from qbt.factor_evaluation import *


class FactorTests(unittest.TestCase):
    def test_open_label(self):
        a=np.array([1,2,3,6,12,24.])[:,None]
        y=forward_open_returns(a,2)
        self.assertAlmostEqual(y[0,0],2.)
        self.assertTrue(np.isnan(y[3:,0]).all())
        self.assertAlmostEqual(forward_open_returns(a,2,1)[0,0],3.)

    def test_missing_exit_no_fill(self):
        a=np.arange(1,9,dtype=float)[:,None];a[4]=np.nan
        self.assertTrue(np.isnan(forward_open_returns(a,2)[1,0]))

    def test_purge(self):
        d=np.array(['2023-12-27','2023-12-28','2023-12-29','2024-01-02','2024-01-03','2024-01-04','2024-01-05'],dtype='datetime64[D]')
        m=split_masks(d,1)
        self.assertTrue(m['train'][0]);self.assertFalse(m['train'][1])
        self.assertTrue(m['validation'][3]);self.assertFalse(m['validation'][-1])

    def test_bh_known_example(self):
        np.testing.assert_allclose(bh_adjust([.01,.04,.03,.002]),[.02,.04,.04,.008])

    def test_hac_reference(self):
        x=np.array([1.,3,2,0,4,2]);z=x-x.mean();n=len(x)
        expected=np.sqrt((sum(z*z)+sum(z[1:]*z[:-1]))/n**2)
        self.assertAlmostEqual(hac_mean(x,1)['se'],expected)

    def test_groups_do_not_use_future_labels(self):
        score=np.arange(100.)[None,:]; y=score.copy();y[0,80:]=np.nan
        f=daily_diagnostics(score,y,np.ones_like(score,bool),min_stocks=10)
        self.assertTrue(np.isnan(f.q5.iloc[0]))
        self.assertEqual(f.q1.iloc[0],9.5)
        self.assertEqual(f.signal_n.iloc[0],100)

    def test_neutralization_removes_control(self):
        x=np.arange(200.)[None,:]
        res=residual_rank(x,[x],np.ones_like(x,bool))
        self.assertLess(np.max(np.abs(res)),1e-12)

    def test_alpha_known_model(self):
        b=np.sin(np.arange(100.))*.01;y=.001+1.5*b
        r=market_regression(y,b)
        self.assertAlmostEqual(r['beta'],1.5)
        self.assertAlmostEqual(r['alpha_annual'],.242)


if __name__=='__main__':unittest.main()
