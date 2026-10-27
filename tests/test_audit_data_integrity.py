"""个人系统常见的静默缓存与行情对齐错误。"""
from pathlib import Path
import sys
import os
import tempfile
import unittest
import zipfile
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qbt.data import FILES, _resolve_source, fingerprint, _PRECISE
from qbt.data import _build_cache, _decode_one, load
from qbt import Calendar


class DataAuditTests(unittest.TestCase):
    def test_loaded_raw_prices_keep_precision_needed_for_exact_limit_orders(self):
        import qbt
        for price,side in ((10.01,'buy'),(9.99,'sell')):
            with self.subTest(price=price,side=side),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);source=root/'source';source.mkdir()
                for name in FILES:(source/(name+'.feather')).write_bytes(name.encode())
                dates=np.array(['2024-01-02','2024-01-03','2024-01-04'],dtype='datetime64[ns]')
                def decode(buf):
                    name=buf.decode()
                    value={'amt':1e8,'num_trades':100.,'cumadj':1.}.get(name,price)
                    return [dict(name='date',type=10),dict(name='600001.SH',type=3)],{
                        'date':dates.copy(),'600001.SH':np.full(3,value)}
                with patch('qbt.data.read_arrow_ipc',side_effect=decode):
                    d=load(source,cache_dir=root/'cache',workers=1,verbose=False)
                def strategy(c):
                    if c.T==1:c.order(d.codes[0],100,limit_price=price if side=='buy' else None)
                    if c.T==2 and side=='sell':c.order(d.codes[0],100,side='sell',limit_price=price)
                free=qbt.Costs(commission=0,min_commission=0,transfer_fee=0,stamp_duty=0)
                r=qbt.run_orders(d,strategy,costs=free)
                self.assertEqual(r.orders[-1].status,'FILLED')
                self.assertEqual(r.order_fills[-1].price,price)
                for name in ('open','high','low','close'):
                    self.assertEqual(getattr(d,name).dtype,np.float64)

    def test_aware_feather_dates_are_not_silently_treated_as_local_session_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture=Path(__file__).parent/'fixtures/audit_arrow/timezone_lz4.arrow'
            with self.assertRaisesRegex(ValueError,'时区'):
                _decode_one(('close',str(fixture),temporary))
            self.assertFalse((Path(temporary)/'close.npy').exists())

    def test_zip_duplicate_panel_names_cannot_silently_mix_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'data.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name in FILES:z.writestr('original/'+name+'.feather',b'old contents')
            target=Path(_resolve_source(archive))
            with zipfile.ZipFile(archive,'a') as z:z.writestr('another_version/open.feather',b'new contents')
            with self.assertRaisesRegex(ValueError,'重复|同名|多个'):_resolve_source(archive)
            self.assertEqual((target/'open.feather').read_bytes(),b'old contents')

    def test_feather_timestamp_units_are_normalized_to_ns(self):
        for unit in ('s','ms','us','ns'):
            with self.subTest(unit=unit),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'open.feather';source.write_bytes(b'dummy')
                dates=np.array(['2024-01-02','2024-01-03'],dtype=f'datetime64[{unit}]')
                fields=[dict(name='date',type=10),dict(name='600001.SH',type=3)]
                columns={'date':dates,'600001.SH':np.array([10.,11.])}
                with patch('qbt.data.read_arrow_ipc',return_value=(fields,columns)):
                    result=_decode_one(('open',str(source),directory))
                np.testing.assert_array_equal(result[3],dates.astype('datetime64[ns]').view('i8'))

    def test_bad_cache_is_rebuilt_instead_of_reusing_its_existing_meta(self):
        for damage in ('missing_panel','malformed_meta','wrong_panel_shape','missing_dates'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source=root/'source';source.mkdir();cache=root/'cache'
                for name in FILES:(source/(name+'.feather')).write_bytes(b'dummy')
                dates=np.array(['2024-01-02','2024-01-03','2024-01-04'],dtype='datetime64[ns]').view('i8')
                calls=[]
                def decode(task):
                    name,_,out=task;calls.append(name)
                    np.save(Path(out)/(name+'.npy'),np.ones((3,1),dtype='float64' if name in _PRECISE else 'float32'))
                    return name,['600001.SH'],(3,1),dates,0.
                with patch('qbt.data._decode_one',side_effect=decode):
                    d=load(source,cache_dir=cache,workers=1,verbose=False)
                    active=Path(d.cache_dir)
                    if damage=='missing_panel':(active/'open.npy').unlink()
                    elif damage=='malformed_meta':(active/'meta.json').write_text('{broken',encoding='utf-8')
                    elif damage=='wrong_panel_shape':np.save(active/'close.npy',np.ones((1,3)))
                    else:(active/'dates.npy').unlink()
                    calls.clear()
                    repaired=load(source,cache_dir=cache,workers=1,verbose=False)
                    self.assertEqual(len(calls),7)
                    self.assertEqual(repaired.close.shape,(3,1))
                    self.assertEqual(repaired.open.shape,(3,1))

    def test_updated_zip_same_size_is_not_silently_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'data.zip'
            def write(payload):
                with zipfile.ZipFile(archive,'w') as z:
                    for name in FILES:z.writestr(name+'.feather',payload)
            write(b'old contents')
            target=Path(_resolve_source(archive))
            write(b'new contents')
            _resolve_source(archive)
            self.assertEqual((target/'open.feather').read_bytes(),b'new contents')

    def test_fingerprint_detects_changed_bytes_even_if_size_mtime_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in FILES:(root/(name+'.feather')).write_bytes(b'old contents')
            before=fingerprint(directory)
            f=root/'close.feather';old=f.stat();f.write_bytes(b'new contents')
            os.utime(f,ns=(old.st_atime_ns,old.st_mtime_ns))
            self.assertNotEqual(fingerprint(directory),before)

    def test_seven_panels_must_have_identical_date_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir()
            for name in FILES:(root/(name+'.feather')).write_bytes(b'dummy')
            def decode(task):
                name,*_=task
                dates=np.array([1,2,3]) if name!='close' else np.array([1,3,2])
                return name,['600001.SH'],(3,1),dates,0.
            with patch('qbt.data._decode_one',side_effect=decode):
                with self.assertRaisesRegex(RuntimeError,'日期'):
                    _build_cache(directory,str(cache),fingerprint(directory),1,False)

    def test_source_change_during_decode_does_not_publish_stale_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';cache.mkdir()
            for name in FILES:(root/(name+'.feather')).write_bytes(b'dummy')
            fp=fingerprint(directory)
            def decode(task):
                name,*_=task
                if name=='close':(root/'close.feather').write_bytes(b'changed')
                return name,['600001.SH'],(3,1),np.array(['2024-01-02','2024-01-03','2024-01-04'],dtype='datetime64[ns]').view('i8'),0.
            with patch('qbt.data._decode_one',side_effect=decode):
                with self.assertRaisesRegex(RuntimeError,'变化'):
                    _build_cache(directory,str(cache),fp,1,False)
            self.assertFalse((cache/'meta.json').exists())

    def test_calendar_rejects_unsorted_duplicate_and_nat_days(self):
        for dates in (['2024-01-03','2024-01-02'],['2024-01-02','2024-01-02'],['2024-01-02','NaT']):
            with self.assertRaises(ValueError):Calendar(np.array(dates,dtype='datetime64[D]'))

    def test_empty_calendar_month_ends_is_empty(self):
        self.assertEqual(Calendar(np.array([],dtype='datetime64[D]')).is_month_end().size,0)

    def test_truncated_calendar_tail_is_not_known_month_end(self):
        cal=Calendar(np.array(['2024-01-30','2024-01-31','2024-02-01','2024-02-02'],dtype='datetime64[D]'))
        np.testing.assert_array_equal(cal.is_month_end(),[False,True,False,False])

    def test_calendar_can_explicitly_confirm_final_trading_month_end(self):
        cal=Calendar(np.array(['2024-03-28','2024-03-29'],dtype='datetime64[D]'))
        np.testing.assert_array_equal(cal.is_month_end(final_month_complete=True),[False,True])

    def test_corrupt_zip_cannot_destroy_previous_extracted_source(self):
        with tempfile.TemporaryDirectory() as directory:
            archive=Path(directory)/'data.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name in FILES:z.writestr(name+'.feather',b'old contents')
            target=Path(_resolve_source(archive))
            with zipfile.ZipFile(archive,'w') as z:
                for name in FILES:z.writestr(name+'.feather',b'new contents')
            contents=archive.read_bytes().replace(b'new contents',b'bad contents',1)
            archive.write_bytes(contents)
            with self.assertRaises(zipfile.BadZipFile):_resolve_source(archive)
            self.assertEqual((target/'open.feather').read_bytes(),b'old contents')


class ActivityInputAudit(unittest.TestCase):
    def test_boolean_features_cannot_be_used_as_price_panels(self):
        from qbt import Dataset,BarDataset
        for cls in (Dataset,BarDataset):
            with self.subTest(dataset=cls.__name__):
                panels={name:np.ones((2,1)) for name in FILES};panels['close']=np.ones((2,1),bool)
                with self.assertRaises(ValueError):
                    cls.from_arrays(['2024-01-02','2024-01-03'],['600001.SH'],**panels)

    def test_feather_boolean_column_is_not_a_one_yuan_stock_quote(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'close.feather';source.write_bytes(b'fixture')
            fields=[dict(name='date',type=10),dict(name='600001.SH',type=6)]
            cols={'date':np.array(['2024-01-02','2024-01-03'],dtype='datetime64[ns]'),
                  '600001.SH':np.array([True,True])}
            with patch('qbt.data.read_arrow_ipc',return_value=(fields,cols)):
                with self.assertRaises(ValueError):_decode_one(('close',str(source),temporary))

    def test_invalid_daily_calendar_does_not_publish_a_completed_cache(self):
        for labels in (['2024-01-02','NaT'],['2024-01-02T09:30','2024-01-02T15:00']):
            with self.subTest(labels=labels),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);cache=root/'cache';cache.mkdir()
                for name in FILES:(root/(name+'.feather')).write_bytes(b'dummy')
                dates=np.array(labels,dtype='datetime64[ns]').view('i8')
                def decode(task):return task[0],['600001.SH'],(2,1),dates,0.
                with patch('qbt.data._decode_one',side_effect=decode):
                    with self.assertRaises((RuntimeError,ValueError)):
                        _build_cache(directory,str(cache),fingerprint(directory),1,False)
                self.assertFalse((cache/'meta.json').exists())

    def test_negative_activity_cannot_contaminate_future_adv(self):
        from qbt import Dataset, BarDataset
        p=np.ones((3,1))
        for name in ('amt','num_trades'):
            panels={key:p.copy() for key in FILES}
            panels[name][0,0]=-1000.
            for dataset in (Dataset, BarDataset):
                with self.subTest(panel=name,dataset=dataset.__name__):
                    with self.assertRaises(ValueError):
                        dataset.from_arrays(np.array(['2024-01-02','2024-01-03','2024-01-04'],dtype='datetime64[D]'),
                                            ['600001.SH'],**panels)

    def test_feather_decode_rejects_bad_activity_and_storage_overflow(self):
        for name,values in (('amt',[1.,-1.]),('num_trades',[1.,-1.]),('amt',[1.,1e100])):
            with self.subTest(panel=name),tempfile.TemporaryDirectory() as directory:
                source=Path(directory)/'source.feather';source.write_bytes(b'dummy')
                fields=[dict(name='date',type=10),dict(name='600001.SH',type=3)]
                columns={'date':np.array(['2024-01-02','2024-01-03'],dtype='datetime64[ns]'),
                         '600001.SH':np.array(values)}
                with patch('qbt.data.read_arrow_ipc',return_value=(fields,columns)):
                    with self.assertRaises(ValueError):_decode_one((name,str(source),directory))
                self.assertFalse((Path(directory)/(name+'.npy')).exists())


if __name__=='__main__':unittest.main(verbosity=2)
