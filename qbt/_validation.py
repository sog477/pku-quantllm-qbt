"""小型公共输入契约：不能让缺失值或因子分数悄悄变成选中。"""
import numpy as np
import re


def naive_datetime_input(value, name):
    """公开构造器只接明确的无时区时间；整数先由调用方声明epoch单位。"""
    if getattr(value, 'tz', None) is not None:
        raise ValueError(f'{name}带时区，请先按目标市场显式转换时间/日期标签')
    raw = np.asarray(value)
    if raw.dtype.kind in 'biuf':
        raise ValueError(f'{name}的整数/数值时间单位不明确，请显式转成datetime64后传入')
    if raw.dtype.kind in 'OSU':
        for item in raw.flat:
            if getattr(item, 'tzinfo', None) is not None:
                raise ValueError(f'{name}带时区，请先按目标市场显式转换时间/日期标签')
            if isinstance(item,(str,bytes,np.str_,np.bytes_)):
                text = item.decode() if isinstance(item,(bytes,np.bytes_)) else str(item)
                if re.search(r'(?:[zZ]|[+-]\d{2}:?\d{2})$',text.strip()):
                    raise ValueError(f'{name}带时区偏移，请先按目标市场显式转换')
            elif isinstance(item,(int,float,np.integer,np.floating,bool)):
                raise ValueError(f'{name}不能混入未声明单位的数值日期')
    parsed = raw if raw.dtype.kind == 'M' else np.asarray(raw,dtype='datetime64')
    result = parsed.astype('datetime64[ns]')
    valid = ~np.isnat(parsed)
    if not np.array_equal(result.astype(parsed.dtype)[valid],parsed[valid]):
        raise ValueError(f'{name}超出纳秒时间范围或精度，不能溢出/截断后继续回测')
    return result


def naive_datetime_scalar(value, name, keep_unit=False):
    result = naive_datetime_input(value, name)
    if result.ndim != 0:
        raise ValueError(f'{name}须为单个日期/时间')
    return np.asarray(value,dtype='datetime64')[()] if keep_unit else result[()]


def integer(value, minimum, maximum, name):
    """精确整数契约，禁止int()截断小数、布尔与越界数量。"""
    if np.ndim(value) != 0 or np.asarray(value).dtype.kind == 'b':
        raise ValueError(f'{name}必须是范围内的整数')
    try:
        result = int(value)
    except (ValueError, OverflowError, TypeError) as error:
        raise ValueError(f'{name}必须是范围内的整数') from error
    if result != value or not minimum <= result <= maximum:
        raise ValueError(f'{name}必须是[{minimum},{maximum}]中的整数')
    return result


def boolean_mask(value, name='mask', shape=None):
    """接受 bool 或数值 0/1；NaN、字符串及其他分数须显式转成条件。"""
    a = np.asarray(value)
    if a.dtype.kind != 'b' and (a.dtype.kind not in 'uif' or not np.isin(a, [0, 1]).all()):
        raise ValueError(f'{name} 必须为bool或0/1；请先处理缺失值、把分数转换为选股条件')
    if shape is not None and a.shape != shape:
        raise ValueError(f'{name} 形状须为{shape}，实际为{a.shape}')
    return a.astype(bool, copy=False)
