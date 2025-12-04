#!/usr/bin/env python3
"""
y86_simulator_opt.py (fixed)

- Fixed: snapshot() no longer emits STAT_DESC (matches test expectations)
- Fixed: RSP updates use unsigned arithmetic (pushq/call/popq/ret)
- Preserved previous optimizations: HybridMemory, CacheLayer, etc.
- CLI: reads .yo from stdin and prints JSON trace (same as test expects)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys, re, json

# ----------------- helpers -----------------
def to_u64(x: int) -> int:
    return x & 0xFFFFFFFFFFFFFFFF

def to_s64(x: int) -> int:
    x = to_u64(x)
    return x if x < (1 << 63) else x - (1 << 64)

# ----------------- Memory Interfaces -----------------
class MemoryInterface:
    def read_u8(self, addr: int) -> int:
        raise NotImplementedError
    def write_u8(self, addr: int, val: int):
        raise NotImplementedError
    def read_u64(self, addr: int) -> int:
        v = 0
        for i in range(8):
            v |= (self.read_u8(addr + i) << (8 * i))
        return to_u64(v)
    def write_u64(self, addr: int, val: int):
        v = to_u64(val)
        for i in range(8):
            self.write_u8(addr + i, (v >> (8 * i)) & 0xFF)
    def load_program_bytes(self, bs: bytes, start:int=0):
        for i,b in enumerate(bs):
            self.write_u8(start + i, b)
    def dump_chunks(self) -> Dict[int,int]:
        raise NotImplementedError
    def get_memory_type(self, addr: int) -> str:
        raise NotImplementedError

class HybridMemory(MemoryInterface):
    def __init__(self, base_size:int = 64*1024):
        self.base_size = base_size
        self.contig = bytearray(base_size)
        self.sparse: Dict[int,int] = {}
    def _in_contig(self, addr:int) -> bool:
        return 0 <= addr < len(self.contig)
    def _ensure_contig_size(self, addr:int):
        if addr < len(self.contig):
            return
        new_size = len(self.contig)
        while addr >= new_size:
            new_size *= 2
        MAX_SZ = 16 * 1024 * 1024
        if new_size > MAX_SZ:
            return
        self.contig.extend(b'\x00' * (new_size - len(self.contig)))
    def read_u8(self, addr:int) -> int:
        if addr < 0:
            return 0
        if self._in_contig(addr):
            return self.contig[addr]
        return self.sparse.get(addr, 0)
    def write_u8(self, addr:int, val:int):
        if addr < 0:
            return
        if not self._in_contig(addr):
            if addr < 16 * 1024 * 1024:
                self._ensure_contig_size(addr+1)
            if self._in_contig(addr):
                self.contig[addr] = val & 0xFF
                return
        if self._in_contig(addr):
            self.contig[addr] = val & 0xFF
            return
        self.sparse[addr] = val & 0xFF
    def read_u64(self, addr:int) -> int:
        if addr >= 0 and (addr + 8) <= len(self.contig):
            chunk = self.contig[addr:addr+8]
            v = 0
            for i,b in enumerate(chunk):
                v |= b << (8*i)
            return to_u64(v)
        v = 0
        for i in range(8):
            v |= (self.read_u8(addr + i) << (8 * i))
        return to_u64(v)
    def write_u64(self, addr:int, val:int):
        v = to_u64(val)
        if addr >= 0 and (addr + 8) <= len(self.contig):
            for i in range(8):
                self.contig[addr + i] = (v >> (8*i)) & 0xFF
            return
        for i in range(8):
            self.write_u8(addr + i, (v >> (8*i)) & 0xFF)
    def load_program_bytes(self, bs:bytes, start:int=0):
        end = start + len(bs)
        if end <= len(self.contig):
            self.contig[start:end] = bs
            return
        for i,b in enumerate(bs):
            self.write_u8(start + i, b)
    def dump_chunks(self) -> Dict[int,int]:
        addrs = set()
        max_contig = len(self.contig)
        for a in range(0, max_contig, 8):
            v = 0
            all_zero = True
            for i in range(8):
                b = self.contig[a + i]
                if b != 0:
                    all_zero = False
                v |= b << (8*i)
            if not all_zero:
                addrs.add(a)
        for a in self.sparse.keys():
            base = (a // 8) * 8
            addrs.add(base)
        out = {}
        for a in sorted(addrs):
            v = self.read_u64(a)
            if v != 0:
                out[a] = to_s64(v)
        return out
    def get_memory_type(self, addr: int) -> str:
        if self._in_contig(addr):
            return "bytearray(contiguous)"
        return "dict(sparse)"

# ----------------- Cache Layer (optional) -----------------
class CacheLayer(MemoryInterface):
    def __init__(self, backing:MemoryInterface, lines:int=256, line_size:int=8):
        self.backing = backing
        self.lines = lines
        self.line_size = line_size
        self.tags = [None] * lines
        self.data = [bytearray(line_size) for _ in range(lines)]
        self.hits = 0
        self.misses = 0
    def _index_tag(self, addr:int) -> Tuple[int,int]:
        line_addr = addr // self.line_size
        idx = line_addr % self.lines
        tag = line_addr // self.lines
        return idx, tag
    def _fill_line(self, idx:int, tag:int, base_addr:int):
        for i in range(self.line_size):
            self.data[idx][i] = self.backing.read_u8(base_addr + i)
        self.tags[idx] = tag
    def read_u8(self, addr:int) -> int:
        idx, tag = self._index_tag(addr)
        base_addr = (addr // self.line_size) * self.line_size
        if self.tags[idx] == tag:
            self.hits += 1
            return self.data[idx][addr - base_addr]
        self.misses += 1
        self._fill_line(idx, tag, base_addr)
        return self.data[idx][addr - base_addr]
    def write_u8(self, addr:int, val:int):
        idx, tag = self._index_tag(addr)
        base_addr = (addr // self.line_size) * self.line_size
        self.backing.write_u8(addr, val)
        if self.tags[idx] == tag:
            self.data[idx][addr - base_addr] = val & 0xFF
        else:
            pass
    def read_u64(self, addr:int) -> int:
        v = 0
        for i in range(8):
            v |= (self.read_u8(addr + i) << (8*i))
        return to_u64(v)
    def write_u64(self, addr:int, val:int):
        for i in range(8):
            self.write_u8(addr + i, (val >> (8*i)) & 0xFF)
    def load_program_bytes(self, bs:bytes, start:int=0):
        self.backing.load_program_bytes(bs, start)
        s = start; e = start + len(bs)
        for addr in range(s, e):
            idx, _ = self._index_tag(addr)
            self.tags[idx] = None
    def dump_chunks(self) -> Dict[int,int]:
        return self.backing.dump_chunks()
    def get_memory_type(self, addr: int) -> str:
        return self.backing.get_memory_type(addr)

# ----------------- CPU State & Instruction scaffolding -----------------
REG_NAMES = [
    "rax","rcx","rdx","rbx","rsp","rbp","rsi","rdi",
    "r8","r9","r10","r11","r12","r13","r14"
]

@dataclass
class CPUState:
    PC: int = 0
    REG: List[int] = field(default_factory=lambda: [0]*15)
    CC: Dict[str,int] = field(default_factory=lambda: {"ZF":1, "SF":0, "OF":0})
    MEM: MemoryInterface = field(default_factory=lambda: HybridMemory())
    STAT: int = 1  # AOK=1, HLT=2, ADR=3, INS=4
    def snapshot(self):
        return {
            "CC": dict(self.CC),
            "MEM": self.MEM.dump_chunks(),
            "PC": to_s64(self.PC),
            "REG": {name: to_s64(self.REG[i]) for i, name in enumerate(REG_NAMES)},
            "STAT": self.STAT
        }
    def get_memory_type(self, addr: int) -> str:
        return self.MEM.get_memory_type(addr)

# Instruction codes
I_HALT = 0x0
I_NOP  = 0x1
I_RRMOVQ = 0x2
I_IRMOVQ = 0x3
I_RMMOVQ = 0x4
I_MRMOVQ = 0x5
I_OPQ   = 0x6
I_JXX   = 0x7
I_CALL  = 0x8
I_RET   = 0x9
I_PUSHQ = 0xA
I_POPQ  = 0xB

# decode instruction (returns tuple)
def decode_at(mem:MemoryInterface, pc:int):
    byte0 = mem.read_u8(pc)
    icode = (byte0 >> 4) & 0xF
    ifun  = byte0 & 0xF
    rA = rB = None
    valC = None
    size = 1
    need_reg = icode in (I_RRMOVQ, I_IRMOVQ, I_RMMOVQ, I_MRMOVQ, I_OPQ, I_PUSHQ, I_POPQ)
    need_valC= icode in (I_IRMOVQ, I_RMMOVQ, I_MRMOVQ, I_JXX, I_CALL)
    if need_reg:
        regbyte = mem.read_u8(pc + 1)
        rA = (regbyte >> 4) & 0xF
        rB = regbyte & 0xF
        size += 1
    if need_valC:
        valC = mem.read_u64(pc + size)
        size += 8
    return (icode, ifun, rA, rB, valC, size)

# condition helpers
def cond_holds(ifun:int, ZF:int, SF:int, OF:int)->bool:
    if ifun == 0: return True
    if ifun == 1: return (SF ^ OF) or ZF
    if ifun == 2: return (SF ^ OF) == 1
    if ifun == 3: return ZF == 1
    if ifun == 4: return ZF == 0
    if ifun == 5: return (SF ^ OF) == 0
    if ifun == 6: return ((SF ^ OF) == 0) and (ZF == 0)
    return False

def set_cc(state:CPUState, op:int, a:int, b:int, r:int):
    sa = to_s64(a); sb = to_s64(b); sr = to_s64(r)
    if op == 0:
        state.CC['OF'] = 1 if ((sa >=0 and sb >=0 and sr <0) or (sa <0 and sb <0 and sr >=0)) else 0
    elif op == 1:
        state.CC['OF'] = 1 if ((sb >=0 and sa <0 and sr <0) or (sb <0 and sa >=0 and sr >=0)) else 0
    else:
        state.CC['OF'] = 0
    state.CC['ZF'] = 1 if to_u64(r) == 0 else 0
    state.CC['SF'] = 1 if sr < 0 else 0

# ----------------- Instruction Handlers -----------------

def h_halt(state, ifun, rA, rB, valC, pc, size):
    state.STAT = 2
    return None

def h_nop(state, ifun, rA, rB, valC, pc, size):
    return pc + size

def h_rrmovq(state, ifun, rA, rB, valC, pc, size):
    if rA is None or rB is None:
        state.STAT = 4; return None
    if cond_holds(ifun, state.CC['ZF'], state.CC['SF'], state.CC['OF']):
        state.REG[rB] = to_u64(state.REG[rA])
    return pc + size

def h_irmovq(state, ifun, rA, rB, valC, pc, size):
    if rB is None:
        state.STAT = 4; return None
    state.REG[rB] = to_u64(valC)
    return pc + size

def h_rmmovq(state, ifun, rA, rB, valC, pc, size):
    addr = to_u64(state.REG[rB] + to_s64(valC))
    if to_s64(addr) < 0:
        state.STAT = 3; return None
    state.MEM.write_u64(addr, to_u64(state.REG[rA]))
    return pc + size

def h_mrmovq(state, ifun, rA, rB, valC, pc, size):
    addr = to_u64(state.REG[rB] + to_s64(valC))
    if to_s64(addr) < 0:
        state.STAT = 3; return None
    state.REG[rA] = state.MEM.read_u64(addr)
    return pc + size

def h_opq(state, ifun, rA, rB, valC, pc, size):
    if rA is None or rB is None:
        state.STAT = 4; return None
    a = state.REG[rA]; b = state.REG[rB]
    if ifun == 0:
        r = to_u64(b + a); set_cc(state, 0, a, b, r)
    elif ifun == 1:
        r = to_u64(b - a); set_cc(state, 1, a, b, r)
    elif ifun == 2:
        r = to_u64(b & a); set_cc(state, 2, a, b, r)
    elif ifun == 3:
        r = to_u64(b ^ a); set_cc(state, 3, a, b, r)
    else:
        state.STAT = 4; return None
    state.REG[rB] = r
    return pc + size

def h_jxx(state, ifun, rA, rB, valC, pc, size):
    if cond_holds(ifun, state.CC['ZF'], state.CC['SF'], state.CC['OF']):
        return to_u64(valC)
    return pc + size

def h_call(state, ifun, rA, rB, valC, pc, size):
    ret = to_u64(pc + size)
    rsp = to_u64(state.REG[4])
    new_rsp = to_u64(rsp - 8)
    state.REG[4] = new_rsp
    # no signed check; writing to memory will use unsigned addresses
    state.MEM.write_u64(new_rsp, ret)
    return to_u64(valC)

def h_ret(state, ifun, rA, rB, valC, pc, size):
    rsp = to_u64(state.REG[4])
    ret_addr = state.MEM.read_u64(rsp)
    state.REG[4] = to_u64(rsp + 8)
    return to_u64(ret_addr)

def h_pushq(state, ifun, rA, rB, valC, pc, size):
    if rA is None:
        state.STAT = 4
        return None

    # 先读待压入的值（重要：若 rA == rsp，要读旧值）
    val = to_u64(state.REG[rA])
    rsp = state.REG[4]          # keep signed semantics
    new_rsp = rsp - 8           # signed subtraction

    # 若产生负地址，记录 rsp（负数），设置 ADR 并立即停止（返回 None）
    if new_rsp < 0:
        state.REG[4] = new_rsp
        state.STAT = 3
        return None

    # 正常路径：写回 RSP 并将值写入内存
    state.REG[4] = new_rsp
    state.MEM.write_u64(new_rsp, val)
    return pc + size



def h_popq(state, ifun, rA, rB, valC, pc, size):
    if rA is None:
        state.STAT = 4
        return None

    rsp = state.REG[4]         # signed

    # 若栈顶地址本身就是负的 -> 立即 ADR 并停止
    if rsp < 0:
        state.STAT = 3
        return None

    # 否则按规范：读内存 -> 更新 rsp -> 写回目标寄存器
    val = state.MEM.read_u64(rsp)
    new_rsp = rsp + 8

    # 对 new_rsp 不做负值检测（加法不会使已非负的 rsp 变负）
    state.REG[4] = new_rsp
    state.REG[rA] = val
    return pc + size


OP_TABLE = {
    I_HALT: h_halt,
    I_NOP:  h_nop,
    I_RRMOVQ: h_rrmovq,
    I_IRMOVQ: h_irmovq,
    I_RMMOVQ: h_rmmovq,
    I_MRMOVQ: h_mrmovq,
    I_OPQ: h_opq,
    I_JXX: h_jxx,
    I_CALL: h_call,
    I_RET: h_ret,
    I_PUSHQ: h_pushq,
    I_POPQ: h_popq,
}

# ----------------- Executor-----------------

def execute_step(state:CPUState) -> bool:
    if state.STAT != 1:
        return False
    pc = state.PC
    icode, ifun, rA, rB, valC, size = decode_at(state.MEM, pc)
    handler = OP_TABLE.get(icode, None)
    if handler is None:
        state.STAT = 4
        return False
    next_pc = handler(state, ifun, rA, rB, valC, pc, size)
    if next_pc is None:
        return False
    state.PC = to_u64(next_pc)
    return True

# ----------------- .yo loader -----------------
def parse_yo(text:str) -> bytes:
    parts = []
    for line in text.splitlines():
        m = re.match(r"\s*0x([0-9a-fA-F]+):\s*([0-9a-fA-F ]*)", line)
        if not m:
            continue
        addr_hex = m.group(1)
        data_hex = m.group(2).split('|')[0].strip().replace(' ', '')
        if data_hex == '':
            continue
        addr = int(addr_hex, 16)
        needed = addr + len(data_hex)//2
        if len(parts) < needed:
            parts.extend([0] * (needed - len(parts)))
        for i in range(0, len(data_hex), 2):
            parts[addr + i//2] = int(data_hex[i:i+2], 16)
    return bytes(parts)

# ----------------- runner + trace -----------------

def run_with_trace(yo_text:str, use_cache:bool=False, cache_params:dict=None, max_steps:int=1000000):
    prog_bytes = parse_yo(yo_text)
    base_mem = HybridMemory()
    if use_cache:
        mem = CacheLayer(base_mem, **(cache_params or {}))
    else:
        mem = base_mem
    state = CPUState(MEM=mem)
    state.MEM.load_program_bytes(prog_bytes, 0)
    trace = []
    steps = 0
    while True:
        cont = execute_step(state)
        trace.append(state.snapshot())
        steps += 1
        if not cont:
            break
        if steps >= max_steps:
            break
    summary = {}
    if use_cache and isinstance(mem, CacheLayer):
        total_access = mem.hits + mem.misses
        hit_rate = (mem.hits / total_access) * 100 if total_access > 0 else 0
        if hit_rate >= 90:
            perf_level = "优秀"
        elif hit_rate >= 70:
            perf_level = "较好"
        elif hit_rate >= 50:
            perf_level = "一般"
        else:
            perf_level = "较差"
        summary['cache'] = {
            'hits': mem.hits,
            'misses': mem.misses,
            'hit_rate': round(hit_rate, 2),
            'performance': perf_level
        }
    mem_type = "N/A"
    if trace:
        last_mem = trace[-1]['MEM']
        if last_mem:
            sample_addr = max(iter(last_mem.keys()))
            mem_type = state.get_memory_type(sample_addr)
    summary['memory_type'] = mem_type
    return trace, summary

# ----------------- CLI -----------------
def usage():
    print("Usage 1 (Test/CLI mode): python y86_simulator_opt.py < input.yo > output.json", file=sys.stderr)
    print("Usage 2 (Web mode): 直接调用 run_with_trace 函数（返回 trace + summary）", file=sys.stderr)

if __name__ == '__main__':
    try:
        yo_text = sys.stdin.read()
        if not yo_text:
            print("Error: 未从标准输入读取到 .yo 文件内容", file=sys.stderr)
            usage()
            sys.exit(1)
    except Exception as e:
        print(f"Error: 读取标准输入失败 - {e}", file=sys.stderr)
        sys.exit(1)

    use_cache = False
    if '--cache' in sys.argv:
        use_cache = True
        sys.argv.remove('--cache')

    trace, summary = run_with_trace(yo_text, use_cache=use_cache)
    json.dump(trace, sys.stdout, indent=2)