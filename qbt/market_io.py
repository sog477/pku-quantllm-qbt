"""Long-format market bars; source amounts stay in the declared local currency."""
import csv
import io
import hashlib
from pathlib import Path
import numpy as np
from .markets import from_market_arrays, market_profile, local_timestamp


def load_market_csv(path, profile, *, intraday=False):
    """CSV: timestamp,code,currency,open,high,low,close,amount[,trades,cumadj].

    amount is actual traded notional, not share volume. Missing rows remain NaN;
    no OHLC or turnover is invented. With explicit ReferenceData, missing cumadj
    is permitted because share events derive the adjustment basis.
    """
    p=market_profile(profile)
    path=Path(path)
    raw=path.read_bytes()
    rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline='')))
    if not rows: raise ValueError('empty market bar file')
    required={'timestamp','code','currency','open','high','low','close','amount'}
    if not required<=rows[0].keys(): raise ValueError(f'market CSV requires {sorted(required)}')
    codes=sorted({r['code'] for r in rows})
    timestamps=sorted({local_timestamp(r['timestamp'],p) for r in rows})
    if not intraday:
        days=np.array(timestamps).astype('datetime64[D]')
        if len(np.unique(days))!=len(days): raise ValueError('multiple intraday timestamps; set intraday=True')
    shape=(len(timestamps),len(codes)); ci={c:j for j,c in enumerate(codes)}; ti={t:i for i,t in enumerate(timestamps)}
    panels={k:np.full(shape,np.nan) for k in ('open','high','low','close','amt','num_trades','cumadj')}
    seen=set()
    for r in rows:
        if r['currency']!=p.currency: raise ValueError('bar currency disagrees with market profile')
        key=ti[local_timestamp(r['timestamp'],p)],ci[r['code']]
        if key in seen: raise ValueError('duplicate timestamp/instrument bar')
        seen.add(key)
        for name,field in [('open','open'),('high','high'),('low','low'),('close','close'),('amt','amount'),('num_trades','trades'),('cumadj','cumadj')]:
            value=r.get(field)
            panels[name][key]=float(value) if value else (1. if name=='cumadj' else np.nan)
    ds=from_market_arrays(p,timestamps,codes,intraday=intraday,**panels)
    ds.source_provenance=dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest(),currency=p.currency)
    return ds
