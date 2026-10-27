"""真实导入对象与磁盘编辑分离的回放，不自动重载研究员状态。"""
import sys
from pathlib import Path
import tempfile
import types
import unittest
import warnings
import os
import py_compile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import qbt
from qbt.provenance import callable_provenance


class ProvenanceAudit(unittest.TestCase):
    def test_file_strategy_loader_does_not_accept_timestamp_valid_old_pyc(self):
        from run_strategy import load_strategy
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'strategy.py'
            path.write_text('def strategy(ctx):\n    return 1\n',encoding='utf-8')
            py_compile.compile(str(path),doraise=True)
            old_stat=path.stat()
            fn,_=load_strategy(str(path)+':strategy');self.assertEqual(fn(None),1)
            path.write_text('def strategy(ctx):\n    return 2\n',encoding='utf-8')
            os.utime(path,ns=(old_stat.st_atime_ns,old_stat.st_mtime_ns))
            updated,_=load_strategy(str(path)+':strategy')
            self.assertEqual(updated(None),2)

    def test_normal_loaded_entry_is_not_reported_stale(self):
        self.assertTrue(callable_provenance(qbt.run)['source_match'])
        self.assertTrue(callable_provenance(qbt.run_orders)['source_match'])

    def test_same_file_edited_while_old_function_remains_loaded_is_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'strategy.py'
            path.write_text('def strategy(ctx):\n    return 1\n',encoding='utf-8')
            namespace={};exec(compile(path.read_bytes(),str(path),'exec'),namespace)
            old=namespace['strategy'];first=callable_provenance(old)
            self.assertTrue(first['source_match'])
            path.write_text('def strategy(ctx):\n    return 2\n',encoding='utf-8')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');second=callable_provenance(old)
            self.assertFalse(second['source_match'])
            self.assertEqual(first['loaded_code_sha256'],second['loaded_code_sha256'])
            self.assertNotEqual(second['loaded_code_sha256'],second['disk_code_sha256'])
            self.assertTrue(caught)
            self.assertEqual(old(None),1)
            fresh={};exec(compile(path.read_bytes(),str(path),'exec'),fresh)
            self.assertTrue(callable_provenance(fresh['strategy'])['source_match'])

    def test_same_name_different_notebook_functions_have_different_identity(self):
        a={};b={}
        exec(compile('def strategy(ctx):\n    return 1','<notebook>','exec'),a)
        exec(compile('def strategy(ctx):\n    return 2','<notebook>','exec'),b)
        pa,pb=callable_provenance(a['strategy']),callable_provenance(b['strategy'])
        self.assertNotEqual(pa['loaded_code_sha256'],pb['loaded_code_sha256'])
        self.assertIsNone(pa['source_match'])


if __name__=='__main__':unittest.main(verbosity=2)
