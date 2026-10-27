"""已加载Python代码的身份；磁盘文件名不等于实际执行物。只记录哈希，不执行源码。"""
import hashlib
import json
from pathlib import Path
import sys
import types
import warnings


def _digest(code):
    # marshal可因字符串驻留/引用表状态而对相等CodeType产生不同bytes。
    # 使用显式、带类型的结构，避免把正常导入误判为旧执行物。
    def constant(value):
        if isinstance(value,types.CodeType):return signature(value)
        if isinstance(value,(tuple,frozenset)):
            items=[constant(v) for v in value]
            if isinstance(value,frozenset):items.sort(key=lambda x:json.dumps(x,sort_keys=True))
            return [type(value).__name__,items]
        return [type(value).__name__,repr(value)]
    def signature(value):
        return dict(bytecode=value.co_code.hex(),constants=[constant(v) for v in value.co_consts],
            names=value.co_names,varnames=value.co_varnames,freevars=value.co_freevars,cellvars=value.co_cellvars,
            argcount=value.co_argcount,posonly=value.co_posonlyargcount,kwonly=value.co_kwonlyargcount,
            flags=value.co_flags,exceptiontable=value.co_exceptiontable.hex())
    encoded=json.dumps(signature(code),sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    return hashlib.sha256(b'qbt-code-v1|'+encoded).hexdigest()


def callable_provenance(callable_):
    code=getattr(callable_,'__code__',None)
    if code is None:
        code=getattr(getattr(callable_,'__call__',None),'__code__',None)
    if code is None:
        return dict(kind=type(callable_).__name__,loaded_code_sha256=None,source_match=None)
    result=dict(kind='python_callable',qualname=code.co_qualname,filename=code.co_filename,
                loaded_code_sha256=_digest(code),python=sys.version.split()[0],source_match=None)
    path=Path(code.co_filename)
    if not path.is_file():
        return result  # notebook/exec代码没有磁盘文件，仍保留实际执行code object身份。
    try:
        source=path.read_bytes()
        result['source_file_sha256']=hashlib.sha256(source).hexdigest()
        module=compile(source,code.co_filename,'exec',dont_inherit=True,optimize=sys.flags.optimize)
        candidates=[]
        def visit(parent):
            for value in parent.co_consts:
                if isinstance(value,types.CodeType):
                    if value.co_qualname==code.co_qualname:candidates.append(value)
                    visit(value)
        visit(module)
        if len(candidates)==1:
            result['disk_code_sha256']=_digest(candidates[0])
            result['source_match']=result['disk_code_sha256']==result['loaded_code_sha256']
        else:
            result['source_check']='ambiguous_or_dynamic_definition'
    except (OSError,SyntaxError,ValueError) as error:
        result['source_check']=type(error).__name__
    if result['source_match'] is False:
        warnings.warn(f"磁盘源码与已加载代码不一致：{code.co_qualname}；请确认是否有意使用旧对象，或重载模块/重启kernel",RuntimeWarning,stacklevel=2)
    return result
