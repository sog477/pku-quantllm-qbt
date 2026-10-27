"""零依赖读取 Apache Arrow IPC / Feather V2 文件。

课程LZ4文件只需要numpy。ZSTD可使用zstandard或pyarrow可选解压后端。

课程的 7 个 .feather 文件是 Arrow IPC 文件格式，record batch 的 buffer 用
LZ4_FRAME 压缩，所以这里自带一个纯 Python 的 LZ4 frame 解码器。

本模块解读平面数值/日期/布尔IPC列，不冒充支持全部Arrow类型；不支持的类型
明确报错。返回普通numpy数组。

格式备忘（踩过的坑）：
  * 文件结构 = b"ARROW1" + 若干 Message + Footer + footer长度(int32) + b"ARROW1"
  * 每个 Message = 0xFFFFFFFF + metadata长度(int32) + flatbuffer + 对齐填充 + body
  * Footer.recordBatches 是 [Block]，而 **Block 是 struct 不是 table**
    （定长 24 字节：long offset + int32 metaDataLength + long bodyLength），
    当成 table 去走 vtable 会读到越界。
  * 压缩打开时，每个 buffer 就是一整个独立的 LZ4 frame（不是裸 block）。
"""

from __future__ import annotations

import struct

import numpy as np

__all__ = ["read_arrow_ipc", "read_feather", "ArrowError", "schema_of"]

_MAGIC = b"ARROW1"
_LZ4_MAGIC = b"\x04\x22\x4d\x18"

# Arrow Type 联合体 tag（见 Arrow 的 Schema.fbs）
_T_NULL = 1
_T_INT = 2
_T_FLOAT = 3
_T_BINARY = 4
_T_UTF8 = 5
_T_BOOL = 6
_T_DECIMAL = 7
_T_DATE = 8
_T_TIME = 9
_T_TIMESTAMP = 10
_T_DURATION = 18


class ArrowError(RuntimeError):
    """文件不是合法的 Arrow IPC，或用了本模块不支持的编码。"""


# --------------------------------------------------------------------------
# flatbuffers 最小导航器
#
# 只需要读，所以不生成代码，按 vtable 直接寻址。所有 `field(i)` 里的 i 都是
# schema 里字段的声明序号（union 会占两个槽：tag 和 value）。
# --------------------------------------------------------------------------
class _FB:
    __slots__ = ("b", "p")

    def __init__(self, buf, pos):
        self.b = buf
        self.p = pos

    def _i16(self, o):
        return struct.unpack_from("<h", self.b, o)[0]

    def _u16(self, o):
        return struct.unpack_from("<H", self.b, o)[0]

    def _i32(self, o):
        return struct.unpack_from("<i", self.b, o)[0]

    def _u32(self, o):
        return struct.unpack_from("<I", self.b, o)[0]

    def root(self):
        """flatbuffer 起点 -> 根表。"""
        return _FB(self.b, self.p + self._u32(self.p))

    def field(self, fid):
        """返回字段的绝对偏移，字段不存在返回 None。"""
        vt = self.p - self._i32(self.p)          # 表头存的是到 vtable 的负偏移
        vt_size = self._u16(vt)
        slot = 4 + fid * 2
        if slot >= vt_size:
            return None
        off = self._u16(vt + slot)
        return (self.p + off) if off else None

    def has(self, fid):
        return self.field(fid) is not None

    def i64(self, fid, default=0):
        o = self.field(fid)
        return struct.unpack_from("<q", self.b, o)[0] if o is not None else default

    def i32(self, fid, default=0):
        o = self.field(fid)
        return self._i32(o) if o is not None else default

    def i16(self, fid, default=0):
        o = self.field(fid)
        return self._i16(o) if o is not None else default

    def u8(self, fid, default=0):
        o = self.field(fid)
        return self.b[o] if o is not None else default

    def table(self, fid):
        o = self.field(fid)
        return _FB(self.b, o + self._u32(o)) if o is not None else None

    def string(self, fid):
        o = self.field(fid)
        if o is None:
            return None
        p = o + self._u32(o)
        return self.b[p + 4:p + 4 + self._u32(p)].decode("utf-8")

    def vector(self, fid):
        """返回 (第一个元素的偏移, 元素个数)。"""
        o = self.field(fid)
        if o is None:
            return 0, 0
        p = o + self._u32(o)
        return p + 4, self._u32(p)

    def vec_table(self, fid, i):
        st, _ = self.vector(fid)
        o = st + i * 4
        return _FB(self.b, o + self._u32(o))


# --------------------------------------------------------------------------
# LZ4 frame 解码（纯 Python）
# --------------------------------------------------------------------------
def _lz4_block(src, out):
    """解一个 LZ4 block，结果 append 到 out（bytearray）。

    试过"预分配 + memoryview 切片赋值"的写法，在这批数据上反而更慢
    （14.2s vs 9.5s / 71MB）：CPython 的 bytearray `+=` 有均摊扩容，比
    memoryview 的 __setitem__ 少一层缓冲协议转换。所以就保持这个写法。
    """
    i = 0
    n = len(src)
    while i < n:
        token = src[i]
        i += 1

        # 字面量长度：高 4 位是 15 表示还有后续字节
        lit_len = token >> 4
        if lit_len == 15:
            while True:
                b = src[i]
                i += 1
                lit_len += b
                if b != 255:
                    break
        if lit_len:
            out += src[i:i + lit_len]
            i += lit_len

        if i >= n:                      # 最后一个序列只有字面量，没有 match
            break

        off = src[i] | (src[i + 1] << 8)
        i += 2
        if off == 0:
            raise ArrowError("LZ4: match offset 为 0")

        ml = token & 0x0F
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4                         # minmatch

        start = len(out) - off
        if start < 0:
            raise ArrowError("LZ4: match 越界")
        if off >= ml:
            out += out[start:start + ml]
        else:
            # 重叠拷贝（run-length 那一类），用一个周期重复出来
            chunk = bytes(out[start:])
            out += (chunk * (ml // off + 1))[:ml]
    return out


def lz4_frame_decompress(buf, expected=None):
    """解一个完整的 LZ4 frame。

    给了 expected 就顺手校验解出来的长度正好对上——这是防"悄悄读错"的
    最后一道闸门，成本几乎为零。
    """
    if buf[:4] != _LZ4_MAGIC:
        raise ArrowError("不是 LZ4 frame（magic 不匹配）")
    flg = buf[4]
    i = 6
    if flg & 0x08:                      # 带 content size
        i += 8
    if flg & 0x01:                      # 带 dict id
        i += 4
    i += 1                              # header checksum

    out = bytearray()
    n = len(buf)
    while i + 4 <= n:
        bsize = struct.unpack_from("<I", buf, i)[0]
        i += 4
        if bsize == 0:                  # end mark
            break
        if bsize & 0x80000000:          # 该 block 未压缩
            raw = bsize & 0x7FFFFFFF
            out += buf[i:i + raw]
            i += raw
        else:
            _lz4_block(memoryview(buf)[i:i + bsize], out)
            i += bsize

    if expected is not None and len(out) != expected:
        raise ArrowError(f"LZ4 frame 解出 {len(out)} 字节，期望 {expected}")
    return out


def _zstd_decompress(buf, expected):
    try:
        import zstandard
    except ImportError as exc:
        try:
            import pyarrow as pa
        except ImportError:
            raise ArrowError(
                "文件用了 ZSTD 压缩，但环境里没有 zstandard或pyarrow；"
                "请安装其中一个，或将源文件导出为LZ4/未压缩IPC"
            ) from exc
        if not pa.Codec.is_available('zstd'):
            raise ArrowError('当前pyarrow没有ZSTD后端，请使用zstandard或重新导出源文件')
        return pa.Codec('zstd').decompress(buf,decompressed_size=expected,asbytes=True)
    return zstandard.ZstdDecompressor().decompress(buf, max_output_size=max(expected, 1))


# --------------------------------------------------------------------------
# Arrow IPC
# --------------------------------------------------------------------------
def _parse_footer(buf):
    n = len(buf)
    if n < 12 or buf[:6] != _MAGIC or buf[-6:] != _MAGIC:
        raise ArrowError("不是 Arrow IPC 文件（magic 不匹配）")

    footer_len = struct.unpack_from("<i", buf, n - 10)[0]
    if footer_len <= 0 or footer_len > n - 10:
        raise ArrowError(f"footer 长度异常: {footer_len}")
    footer = _FB(buf, n - 10 - footer_len).root()

    schema = footer.table(1)
    if schema is None:
        raise ArrowError("footer 里没有 schema")

    fields = []
    st, cnt = schema.vector(1)
    for i in range(cnt):
        f = schema.vec_table(1, i)
        node = f.table(3)
        fields.append({
            "name": f.string(0),
            "type": f.u8(2),
            "node": node,
            "timezone": node.string(1) if f.u8(2) == _T_TIMESTAMP and node is not None else None,
            "children": _field_children(f),
        })

    # recordBatches 是 [Block]，Block 是 struct：24 字节定长
    blocks = []
    st, cnt = footer.vector(3)
    for i in range(cnt):
        o = st + i * 24
        blocks.append((
            struct.unpack_from("<q", buf, o)[0],       # offset
            struct.unpack_from("<i", buf, o + 8)[0],   # metaDataLength
            struct.unpack_from("<q", buf, o + 16)[0],  # bodyLength
        ))

    # dictionaries 也可能存在（本课程数据没有），这里只做检查
    st, cnt = footer.vector(2)
    if cnt:
        raise ArrowError("暂不支持带 dictionary encoding 的 Arrow 文件")

    return fields, blocks


def _field_children(f):
    """嵌套类型（list/struct）会有子字段，本课程数据是平的，但读取时要能报错。"""
    st, cnt = f.vector(5)
    return cnt


def _arrow_dtype(field):
    """把 Arrow 类型映射成 numpy dtype。返回 (dtype, 类型名)。"""
    t = field["type"]
    n = field["node"]

    if t == _T_FLOAT:
        precision = n.u8(0, 2) if n is not None else 2
        return {0: np.dtype("<f2"), 1: np.dtype("<f4"), 2: np.dtype("<f8")}.get(
            precision, np.dtype("<f8")
        ), "float"
    if t == _T_INT:
        width = n.i32(0, 64) if n is not None else 64
        signed = (n.u8(1, 1) if n is not None else 1) != 0
        kind = "i" if signed else "u"
        return np.dtype(f"<{kind}{max(width, 8) // 8}"), "int"
    if t == _T_BOOL:
        return np.dtype("?"), "bool"
    if t == _T_TIMESTAMP:
        unit = n.i16(0, 0) if n is not None else 0
        if unit not in (0,1,2,3):
            raise ArrowError(f'未知Timestamp单位{unit}，不能猜成ns')
        unit = {0: "s", 1: "ms", 2: "us", 3: "ns"}[unit]
        return np.dtype("<i8"), f"ts:{unit}"
    if t == _T_DATE:
        unit = n.i16(0, 1) if n is not None else 1
        if unit not in (0,1):
            raise ArrowError(f'未知Date单位{unit}')
        return (np.dtype('<i4'),'date32') if unit == 0 else (np.dtype('<i8'),'date64')
    if t == _T_TIME:
        width = n.i32(1,32) if n is not None else 32
        if width not in (32,64):
            raise ArrowError(f'未知Time位宽{width}')
        return np.dtype(f'<i{width//8}'), 'int'
    if t == _T_DURATION:
        return np.dtype("<i8"), "int"
    raise ArrowError(f"不支持的 Arrow 类型 tag={t}")


def _decompress(raw, expected, codec, what=""):
    """按IPC压缩前缀解码；expected为逻辑所需最小长度，位图可有多余字节。"""
    if not raw:
        return b""
    if codec == -1:
        return raw
    if raw[:4] == _LZ4_MAGIC:
        decoded = lz4_frame_decompress(raw)  # 兼容旧的裸LZ4帧输入。
    else:
        if len(raw) < 8:
            raise ArrowError(f'buffer{what} 缺少压缩长度前缀')
        ulen = struct.unpack_from('<q',raw,0)[0]
        body = raw[8:]
        if ulen == -1:
            decoded = body  # 压缩批次内这一块按原样存，不得把-1当数据。
        elif ulen < expected:
            raise ArrowError(f'buffer{what} 声明解压长度{ulen}小于逻辑所需{expected}')
        elif codec == 0:
            if body[:4] == _LZ4_MAGIC:
                decoded = lz4_frame_decompress(body,ulen)
            elif len(body) == ulen:
                decoded = body  # 兼容长度前缀后的原样存储。
            else:
                raise ArrowError(f'buffer{what} 不是LZ4 frame')
        elif codec == 1:
            decoded = _zstd_decompress(body,ulen)
            if len(decoded) != ulen:
                raise ArrowError(f'buffer{what} ZSTD长度不符')
        else:
            raise ArrowError(f'未知压缩codec={codec}')
    if len(decoded) < expected:
        raise ArrowError(f'buffer{what} 只有{len(decoded)}字节，逻辑所需{expected}')
    return decoded


def read_arrow_ipc(buf):
    """读 Arrow IPC 文件。

    返回 (fields, columns)：
      fields  —— schema 描述列表
      columns —— {列名: numpy 数组}
    """
    fields, blocks = _parse_footer(buf)
    if not fields:
        return [], {}

    if len({f['name'] for f in fields}) != len(fields):
        raise ArrowError('字典列接口无法表示重复列名，不能覆盖同名列')
    chunks = {f["name"]: [] for f in fields}
    for f in fields:
        if f["children"]:
            raise ArrowError("暂不支持嵌套列类型")

    footer_start = len(buf)-10-struct.unpack_from('<i',buf,len(buf)-10)[0]
    for (off, meta_len, body_len) in blocks:
        if off < 8 or meta_len < 8 or body_len < 0 or off+meta_len+body_len > footer_start:
            raise ArrowError('RecordBatch的元数据/数据体超出文件记录边界')
        if struct.unpack_from("<I", buf, off)[0] != 0xFFFFFFFF:
            raise ArrowError("Message 缺少 continuation 标记")
        if 8+struct.unpack_from('<I',buf,off+4)[0] > meta_len:
            raise ArrowError('Message元数据长度超过Block声明')
        msg = _FB(buf, off + 8).root()
        if msg.u8(1) != 3:
            raise ArrowError('RecordBatch Block的消息类型不符')
        header = msg.table(2)            # Message.header（union value）= RecordBatch
        if header is None:
            raise ArrowError('RecordBatch缺少header，不能跳过整批数据')
        # Block.metaDataLength已经包括消息前缀与对齐，不能额外再加8字节。
        body_off = off + meta_len
        batch_length = header.i64(0,0)

        codec = -1
        comp = header.table(3)           # RecordBatch.compression
        if comp is not None:
            codec = comp.u8(0, 0)
            if codec not in (0,1) or comp.u8(1,0) != 0:
                raise ArrowError('不支持的IPC压缩方式')

        nst, n_nodes = header.vector(1)
        bst, n_bufs = header.vector(2)
        nodes = [
            (struct.unpack_from("<q", buf, nst + i * 16)[0],
             struct.unpack_from("<q", buf, nst + i * 16 + 8)[0])
            for i in range(n_nodes)
        ]
        bufs = [
            (struct.unpack_from("<q", buf, bst + i * 16)[0],
             struct.unpack_from("<q", buf, bst + i * 16 + 8)[0])
            for i in range(n_bufs)
        ]

        for j, f in enumerate(fields):
            # 每列一个 FieldNode，两个 Buffer（有效性位图 + 值）；位图可为 0 长度
            if j >= len(nodes) or 2 * j + 2 > len(bufs):
                raise ArrowError(
                    f"RecordBatch 的 node/buffer 数与 schema 不符（列 {f['name']}，"
                    f"nodes={len(nodes)} bufs={len(bufs)} j={j}）"
                )

            length, null_count = nodes[j]
            if length != batch_length or not 0 <= null_count <= length:
                raise ArrowError(f'列{f["name"]}的行数/空值数量与RecordBatch不符')
            v_off, v_len = bufs[2 * j]
            d_off, d_len = bufs[2 * j + 1]
            if any(offset < 0 or size < 0 or offset+size > body_len
                   for offset,size in ((v_off,v_len),(d_off,d_len))):
                raise ArrowError(f'列{f["name"]}的buffer超出RecordBatch数据体，不能读footer作行情')

            dtype, kind = _arrow_dtype(f)
            expected = (length+7)//8 if kind == 'bool' else length * dtype.itemsize
            raw = _decompress(
                buf[body_off + d_off: body_off + d_off + d_len],
                expected, codec, what=f"({f['name']})",
            )
            if len(raw) < expected:
                raise ArrowError(
                    f"列 {f['name']} 解压后只有 {len(raw)} 字节，期望 {expected}"
                )
            data = (np.unpackbits(np.frombuffer(raw,dtype=np.uint8),bitorder='little')[:length].astype(bool)
                    if kind == 'bool' else np.frombuffer(raw,dtype=dtype,count=length))

            mask = None
            if null_count:
                if v_len == 0:
                    raise ArrowError(f"列 {f['name']} 有 null 但没有有效性位图")
                need = (length + 7) // 8
                vraw = _decompress(
                    buf[body_off + v_off: body_off + v_off + v_len],
                    need, codec, what=f"({f['name']} validity)",
                )
                if len(vraw) < need:
                    raise ArrowError(f"列 {f['name']} 有效性位图长度不足")
                mask = np.unpackbits(
                    np.frombuffer(vraw, dtype=np.uint8), bitorder="little"
                )[:length].astype(bool)
            if kind.startswith("ts:"):
                data = data.view(f"datetime64[{kind[3:]}]").copy()
            elif kind == "date32":
                data = data.astype('datetime64[D]')
            elif kind == 'date64':
                data = data.view('datetime64[ms]').copy()
            if mask is not None:
                if data.dtype.kind == 'M':
                    data[~mask] = np.datetime64('NaT')
                elif kind == 'float':
                    data = np.where(mask,data,np.nan).astype(dtype,copy=False)
                else:
                    data = np.where(mask,data.astype(object),np.nan)
            elif kind == 'float':
                data = data.copy()
            chunks[f['name']].append(data)

    missing = [k for k, v in chunks.items() if not v]
    if missing:
        raise ArrowError(f"这些列没有读到数据: {missing[:5]}")
    out = {name:parts[0] if len(parts)==1 else np.concatenate(parts) for name,parts in chunks.items()}
    return fields, out


def schema_of(buf):
    """只看 schema，不读数据。"""
    fields, _ = _parse_footer(buf)
    return [{"name": f["name"], "type": f["type"]} for f in fields]


def read_feather(path):
    """读一个 .feather / .arrow 文件，返回 {列名: numpy 数组}。"""
    with open(path, "rb") as fh:
        buf = fh.read()
    return read_arrow_ipc(buf)[1]
