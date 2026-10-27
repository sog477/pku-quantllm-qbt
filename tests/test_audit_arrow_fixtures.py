"""独立pyarrow 20生成的真实IPC字节，不用本读取器生成标准答案。"""
from pathlib import Path
import json
import shutil
import struct
import sys
import tempfile
import unittest
import importlib.util
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qbt.arrow import read_arrow_ipc,_decompress,ArrowError,_parse_footer,_FB
from qbt.data import FILES,load

FIXTURES=Path(__file__).parent/'fixtures/audit_arrow'
ORACLE=json.loads((FIXTURES/'oracle.json').read_text(encoding='utf-8'))


class IndependentArrowAudit(unittest.TestCase):
    def test_buffer_offset_cannot_read_footer_metadata_as_stock_prices(self):
        data=bytearray((FIXTURES/'none_9.arrow').read_bytes())
        _,blocks=_parse_footer(data)
        offset,_,body_length=blocks[0]
        message=_FB(data,offset+8).root()
        buffers,_=message.table(2).vector(2)
        # 第3列值区指向RecordBatch体以外，那里是EOS/footer，不是价格。
        struct.pack_into('<q',data,buffers+5*16,body_length)
        with self.assertRaises(ArrowError):read_arrow_ipc(data)

    def test_pyarrow_is_a_working_zstd_fallback_when_zstandard_is_unavailable(self):
        data=(FIXTURES/'zstd_3.arrow').read_bytes()
        with patch.dict(sys.modules,{'zstandard':None}):
            if importlib.util.find_spec('pyarrow'):
                _,columns=read_arrow_ipc(data)
                np.testing.assert_array_equal(columns['600001.SH'],np.asarray(ORACLE['first'],float))
            else:
                with self.assertRaisesRegex(ArrowError,'ZSTD|zstandard'):read_arrow_ipc(data)

    def test_zstd_matches_or_explicitly_reports_missing_dependency(self):
        available=importlib.util.find_spec('zstandard') or importlib.util.find_spec('pyarrow')
        for chunk in (3,9):
            with self.subTest(chunk=chunk):
                data=(FIXTURES/f'zstd_{chunk}.arrow').read_bytes()
                if available:
                    _,columns=read_arrow_ipc(data)
                    np.testing.assert_array_equal(columns['date'],np.array(ORACLE['dates'],dtype='datetime64[ns]'))
                    np.testing.assert_array_equal(columns['600001.SH'],np.asarray(ORACLE['first'],float))
                    np.testing.assert_array_equal(columns['000001.SZ'],ORACLE['second'])
                else:
                    with self.assertRaisesRegex(ArrowError,'ZSTD|zstandard'):read_arrow_ipc(data)

    def test_record_batches_without_nulls_are_not_silently_truncated(self):
        _,columns=read_arrow_ipc((FIXTURES/'nonull_lz4_3.arrow').read_bytes())
        np.testing.assert_array_equal(columns['600001.SH'],np.arange(10.,19.))
        np.testing.assert_array_equal(columns['date'],np.array(ORACLE['dates'],dtype='datetime64[ns]'))

    def test_single_and_multi_batch_compressed_and_uncompressed_match_oracle(self):
        for codec in ('none','lz4'):
            for chunk in (3,9):
                with self.subTest(codec=codec,chunk=chunk):
                    _,columns=read_arrow_ipc((FIXTURES/f'{codec}_{chunk}.arrow').read_bytes())
                    np.testing.assert_array_equal(columns['date'],np.array(ORACLE['dates'],dtype='datetime64[ns]'))
                    np.testing.assert_array_equal(columns['600001.SH'],np.asarray(ORACLE['first'],float))
                    np.testing.assert_array_equal(columns['000001.SZ'],ORACLE['second'])

    def test_dates_timestamp_and_boolean_nulls_keep_their_meaning(self):
        for codec in ('none','lz4'):
            with self.subTest(codec=codec):
                _,columns=read_arrow_ipc((FIXTURES/f'types_{codec}.arrow').read_bytes())
                for name in ('timestamp_ms','date32','date64'):
                    np.testing.assert_array_equal(columns[name].astype('datetime64[D]'),
                        np.array(['2024-01-02','NaT','2024-01-04'],dtype='datetime64[D]'))
                self.assertEqual(columns['flag'][:2].tolist(),[True,False])
                self.assertTrue(np.isnan(columns['flag'][2]))
                self.assertEqual(columns['count'][[0,2]].tolist(),[1,3])
                self.assertTrue(np.isnan(columns['count'][1]))

    def test_compression_minus_one_means_this_buffer_is_uncompressed(self):
        for codec in (0,1):
            self.assertEqual(_decompress(struct.pack('<q',-1)+b'abc',3,codec),b'abc')

    def test_seven_aligned_files_cannot_all_silently_lose_later_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source';source.mkdir()
            for name in FILES:shutil.copyfile(FIXTURES/'nonull_lz4_3.arrow',source/(name+'.feather'))
            d=load(source,cache_dir=Path(temporary)/'cache',workers=1,verbose=False)
            self.assertEqual(d.T,9)
            np.testing.assert_array_equal(d.close[:,0],np.arange(10.,19.))


if __name__=='__main__':unittest.main(verbosity=2)
