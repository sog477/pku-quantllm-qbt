"""早期网页摘要中的因子研究/统计验证风险，不冒充完整外部案例。"""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qbt.factor_evaluation import market_regression


class StatisticalAuditTests(unittest.TestCase):
    def test_hac_does_not_compress_missing_trading_days(self):
        t=np.arange(100.)
        b=np.sin(t)*.01
        y=.001+.5*b+np.cos(t*.2)*.005
        y[::3]=np.nan
        valid=np.isfinite(y)
        X=np.column_stack([np.ones(valid.sum()),b[valid]])
        coef=np.linalg.lstsq(X,y[valid],rcond=None)[0]
        z=np.zeros((len(y),2));z[valid]=X*(y[valid]-X@coef)[:,None]
        # lag1 Bartlett权重1/2；相隔缺失交易日的残差不应被拼成相邻观测。
        S=z.T@z+.5*(z[1:].T@z[:-1]+z[:-1].T@z[1:])
        bread=np.linalg.pinv(X.T@X)
        se=np.sqrt(np.diag(bread@S@bread))
        r=market_regression(y,b,lag=1)
        self.assertAlmostEqual(r['alpha_upper'],(coef[0]+1.96*se[0])*242,places=12)

    def test_unidentifiable_beta_is_not_reported_as_zero(self):
        r=market_regression(np.array([.001,.002,.003,.004]),np.zeros(4))
        self.assertTrue(np.isnan(r['beta']))
        self.assertTrue(np.isnan(r['alpha_p']))

    def test_two_observations_do_not_claim_alpha_significance(self):
        r=market_regression(np.array([.001,.003]),np.array([.001,.002]))
        self.assertTrue(np.isnan(r['alpha_p']))
        self.assertTrue(np.isnan(r['alpha_lower']))


if __name__=='__main__':unittest.main(verbosity=2)
