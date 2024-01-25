'''
TODO:

- error handling

Where can free chunks end up:
- tcache (small or moderately large)
- slab (small)
- free extents (large, or small if their extent is freed)
- free extents, coalesced
- quarantine (in CheriBSD, if caprevoke is enabled)
'''

from enum import Enum
import bisect

class RTreeState(Enum):
    OK = 1       # Valid compact representation of extent
    INVALID = 2  # We cannot even access the level 2 table (for level >= 2)
    NULL = 3     # Location ok but has null there (i.e. no such entry)
    SENTINEL = 4 # Location ok but SC_NSIZES << 1 found
                 # indicates extent is no longer valid due to coalescing


class ChunkState(Enum):
    USED = 1                   # In use
    FREE_TCACHE = 2            # In tcache
    FREE_SLAB = 3              # In slab
    FREE_EXTENT = 4            # In one of the extents in an arena
    FREE_EXTENT_COALESCED = 5  # Same as before, but coalesced
    FREE_QUARANTINE = 6        # In caprevoke quarantine


def read_captags(addr, length):
    STRIDE = 0x10 * 8 # 0x10 (bytes per cap) * 8 (bits) per byte of bitmask
    bmlen = length // STRIDE
    if length % STRIDE: bmlen += 1
    bitmap = ctypes.create_string_buffer(bmlen)
    class ptrace_io_desc(ctypes.Structure):
        _fields_ = [('piod_op', ctypes.c_int),
                    ('piod_offs', ctypes.c_uint64),
                    ('piod_addr', ctypes.c_uint64),
                    ('piod_len', ctypes.c_size_t)]
    desc = ptrace_io_desc()
    PIOD_READ_CHERI_TAGS = 5
    desc.piod_op = PIOD_READ_CHERI_TAGS
    desc.piod_offs = addr
    desc.piod_addr = ctypes.addressof(bitmap)
    desc.piod_len = bmlen
    libc = ctypes.CDLL(None)
    ptrace = libc.syscall
    ptrace.argtypes = (ctypes.c_uint64,) * 4
    # print(ctypes.addressof(desc))
    PTRACE = 26
    PT_IO = 12
    res = ptrace(PTRACE,
                 PT_IO,
                 gdb.selected_inferior().pid,
                 ctypes.addressof(desc),
                 0)
    if res != 0:
        raise RuntimeError
    return bitmap.raw


class GefJemallocManager(GefManager):
    """Class managing session heap."""
    def __init__(self) -> None:
        self.reset_caches()

        self.__tsd_offset = None

        # We do not reset it in reset_caches
        # This is an optimisation; if __prev_maps don't change across cache resets,
        # we don't need to find new heaps
        self.__prev_maps = None
        return

    def reset_caches(self) -> None:
        super().reset_caches()

        # Hackish solution
        # If our function call is originating from hook_stop_handler,
        # we then update the maps, because at some cache resets such as exit or objfile loading,
        # jemalloc is not available
        stk = inspect.stack()
        for f in stk:
            if f.function == 'hook_stop_handler':
                self.update_heap_maps()
                break
            elif f.function == 'exit_handler':
                self.__tsd_offset = None
                self.__prev_maps = None
                break
        return

    def get_tls_offset(self):
        if not self.__tsd_offset:
            # If TLS offset code is not found within first 10 instructions of __malloc,
            # then this is -O0 code, and we'll grab the offset from tsd_get hopefully
            attempts = 10
            addr = 0
            try:
                addr = parse_address("tsd_get")
            except gdb.error:
                addr = parse_address("__malloc")

            func = gdb_disassemble(addr, count = attempts + 1)
            for i in range(attempts):
                insn = next(func)
                if insn.mnemonic == 'adrp' and '0x' in insn.operands[1]:
                    # gdb disassembly bug
                    base = int(insn.operands[1], 16) & ((1 << 32) - 1)
                    insn = next(func)
                    if insn.mnemonic != 'add':
                        raise ValueError('TSD offset failure')
                    offset = int(insn.operands[2].strip('# '), 16)
                    tlsoffset = dereference(base + offset).cast(cached_lookup_type("size_t"))
                    self.__tsd_offset = tlsoffset
                    break
                    
        return self.__tsd_offset

    def _get_rtree_loc(self, ptr):
        i = (ptr >> 30) & 0x3ffff
        j = (ptr >> 12) & 0x3ffff
        if self.rtree_level == 2:
            repr_ = self.rtree['root'][i]['child']['repr']
            if int(repr_) == 0:
                raise ValueError
            arr_ptr = cached_lookup_type('__uintcap_t').pointer()
            metadata = repr_.cast(arr_ptr)[j]
            ull_ptr = cached_lookup_type('unsigned long long').pointer()
            metadata_ull = metadata.address.cast(ull_ptr).dereference() # convenience
            return (metadata, metadata_ull)
        else:
            raise NotImplementedError

    def rtree_status(self, ptr):
        try:
            metadata, metadata_val = self._get_rtree_loc(ptr)
            if metadata_val == 0:
                return RTreeState.INVALID
            sentinel = (jemalloc.idx2sz.type.range()[1]+1) << 1 # Had extent here before, no longer
            if metadata_val == sentinel:
                return RTreeState.SENTINEL
            else:
                if metadata_val - parse_cap_internal(metadata)[2] == sentinel:
                    return RTreeState.SENTINEL
                return RTreeState.OK
        except ValueError:
            return RTreeState.INVALID

    def read_rtree(self, ptr):
        m, mv = self._get_rtree_loc(ptr)
        extent_addr = parse_cap_internal(m)[2]
        return (mv, extent_addr)

    def find_blocks(self):
        try:
            b0 = gdb.parse_and_eval("b0")
            block = b0['blocks']
            blocks = []
            while block != 0:
                blocks.append(block)
                block = block['next']
            return blocks
        except:
            return [] # Haven't got b0 yet, booting of jemalloc not done yet

    def update_heap_maps(self):
        maps = gef.memory.maps
        if not self.__prev_maps or maps != self.__prev_maps:
            blocks = self.find_blocks()
            for sect in maps:
                if sect.is_writable() and sect.path == '':
                    if sect.page_start in blocks:
                        sect.path = '[heap block]'
                    else:
                        try:
                            if self.rtree_status(sect.page_start) == RTreeState.OK:
                                sect.path = '[heap]'
                        except:
                            break
            self.__prev_maps = maps

    @property
    def tsd(self):
        tsd_ret = {}

        tlsoffset = self.get_tls_offset()
        tsd_type = cached_lookup_type('tsd_t').pointer()
        current_thread = gdb.selected_thread()

        for thread in gdb.selected_inferior().threads():
            thread.switch()
            tsd = gdb.parse_and_eval(f"$ctpidr+{tlsoffset}").cast(tsd_type)
            tsd_ret[thread.num] = tsd

        current_thread.switch()

        return tsd_ret
    
    @property
    def free_extents(self):
        # for tsd in jemalloc.tsd.values():
        ext_types = ('extents_dirty', 'extents_muzzy', 'extents_retained')
        ret = []
        for arena in self.arenas:
            for ext_type in ext_types:
                extents = arena[ext_type]
                bitmap = extents['bitmap']
                # Convert 4 64-bit into one big Python int
                actual_bitmap = sum([int(bitmap[i])<<(64*i) for i in range(bitmap.type.range()[1])])
                i = 0
                while actual_bitmap != 0:
                    if actual_bitmap & 1:
                        cnt = extents['nextents'][i]['repr']
                        curr = extents['heaps'][i]['ph_root']
                        for j in range(cnt):
                            ret.append(curr)
                            # print(curr)
                            curr = curr['ph_link']['phn_next']
                    i += 1
                    actual_bitmap >>= 1
        return ret

    def find_in_free_extents(self, addr):
        for extent in self.free_extents:
            if addr >= extent['e_addr']:
                if (addr - int(extent['e_addr'])) < int(extent['e_bsize']):
                    return [extent]
        return []

    @property
    def rtree_level(self):
        return 2 # Unsure how to calculate...yet

    @property
    def arenas(self):
        arena_cnt = int(gdb.parse_and_eval('narenas_total.repr'))
        arenas = gdb.parse_and_eval('__je_arenas')
        arena_t_ptr = cached_lookup_type('arena_t').pointer()
        return [
            arenas[i]['repr'].cast(arena_t_ptr) 
            for i in range(arena_cnt) if arenas[i]['repr'] != 0
        ]

    @property
    def rtree(self):
        return gdb.parse_and_eval('&__je_extents_rtree')

    @property
    def idx2sz(self):
        return gdb.parse_and_eval('&__je_sz_index2size_tab').dereference()

    @property
    def tcache_enabled(self):
        return int(gdb.parse_and_eval('__je_opt_tcache'))
    
    @property
    def small_max(self):
        return int(self.idx2sz[self.small_max_idx])

    @property
    def small_max_idx(self):
        _, small_max_idx = gdb.parse_and_eval('__je_bin_infos').type.range()
        return small_max_idx

    @property
    def max_tcache_idx(self):
        return int(gdb.parse_and_eval('__je_nhbins'))
    
class GefHeapQuarantineManager(GefManager):
    """Class managing heap quarantine."""
    def __init__(self) -> None:
        self.reset_caches()
        return

    def reset_caches(self) -> None:
        super().reset_caches()
        self.__read_chunks()
        return

    def __read_chunks(self) -> None:
        self.__chunks = set() # Set of chunks in quarantine
        self.__has_quarantine = False
        try:
            app_quarantine = gdb.parse_and_eval('application_quarantine')
        except:
            app_quarantine = None
        if app_quarantine:
            self.__has_quarantine = True
            curr = app_quarantine['list']
            while int(curr) != 0:
                num_desc = curr['num_descriptors']
                # for i in range(num_desc):
                #     self.__chunks.add(int(curr['slab'][i]['ptr']))

                # Single read optimisation
                desc_sz = curr['slab'][0].type.sizeof
                data = gef.memory.read(curr['slab'], num_desc * desc_sz)
                self.__chunks = {
                    u64(data[i * desc_sz:i * desc_sz + 8])
                    for i in range(num_desc)
                }
                curr = curr['next']

    @property
    def chunks(self):
        return self.__chunks

    @property
    def has_quarantine(self):
        return self.__has_quarantine
    

jemalloc = GefJemallocManager()
quarantine = GefHeapQuarantineManager()

# Hooking
orig_reset_fn = gef.heap.reset_caches
def hooked_reset():
    jemalloc.reset_caches()
    quarantine.reset_caches()
    orig_reset_fn()
gef.heap.reset_caches = hooked_reset
gef.memory.read_captags = read_captags # hack


class JemallocChunk:
    def __init__(self, addr: int) -> None:
        self.addr = addr

        # Cached so that repeated calls by other methods don't do repeated redundant work
        self.__info = None
        self.__status = None
        self.init_info()
        self.get_chunk_status()
        return

    def init_info(self):
        if jemalloc.rtree_level == 2:
            status = jemalloc.rtree_status(self.addr)
            if status == RTreeState.INVALID or status == RTreeState.NULL:
                raise ValueError
            elif status == RTreeState.OK:
                metadata_val, extent = jemalloc.read_rtree(self.addr)
                szind = (metadata_val - extent) >> 1
                actual_extent = gdb.Value(extent).cast(cached_lookup_type('extent_t').pointer())
                sz = int(jemalloc.idx2sz[szind])
                region_idx = (self.addr - int(actual_extent['e_addr'])) // sz
                base_addr = region_idx * sz + int(actual_extent['e_addr'])
            else:
                self.__status = ChunkState.FREE_EXTENT_COALESCED
                res = jemalloc.find_in_free_extents(self.addr)
                if len(res) != 0:
                    actual_extent = res[0]
                    base_addr = int(actual_extent['e_addr'])
                    sz = int(actual_extent['e_bsize'])
                    szind = region_idx = 0xffff # Sentinel
                else:
                    raise ValueError

            self.__info = {
                'size': sz,               # total usable chunk size
                'szind': szind,           # index of size
                'extent': actual_extent,  # the extent this chunk belongs to
                'base_addr': base_addr,   # where this chunk starts, addr can be base_addr+off
                'region_idx': region_idx, # which chunk in the extent is this chunk?
                'status': None
            }
        else:
            raise NotImplementedError

    def in_tcache(self) -> bool:
        TCACHE_STR = 'cant_access_tsd_items_directly_use_a_getter_or_setter_tcache'
        if not jemalloc.tcache_enabled or self.szind >= jemalloc.max_tcache_idx:
            return False
        for tsd in jemalloc.tsd.values():
            if self.is_small():
                # if big, use bins_large, and check if oob
                bin_ = tsd[TCACHE_STR]['bins_small'][self.szind]
            else:
                bin_ = tsd[TCACHE_STR]['bins_large'][self.szind - jemalloc.small_max_idx]
            ncached = int(bin_['ncached'])
            sz = ncached * cached_lookup_type('void').pointer().sizeof
            start = int(bin_['avail']) - sz
            avail = gef.memory.read(start, sz)
            if p64(self.base_addr) in avail:
                return True
        return False

    # In my slab, am I free?
    def free_in_slab(self) -> bool:
        idx = self.region_idx
        i = idx // 64
        j = idx % 64
        bitmap = int(self.extent['e_slab_data']['bitmap'][i])
        return (bitmap & (1 << j)) != 0
    
    def get_chunk_status_(self):
        if self.base_addr in quarantine.chunks:
            return ChunkState.FREE_QUARANTINE
        if self.in_tcache():
            return ChunkState.FREE_TCACHE
        if self.is_small():
            if self.free_in_slab():
                return ChunkState.FREE_SLAB
        else:
            # 1 for dirty, 2 for muzzy, 3 for retained
            # all 3 mean not in use, see extent_state_t
            # in extent_structs.h
            if (int(self.extent['e_bits']) >> 16) & 0b11 > 0:
                return ChunkState.FREE_EXTENT
        return ChunkState.USED

    def get_chunk_status(self):
        if not self.__info['status']:
            self.__info['status'] = self.get_chunk_status_()
        return self.__info['status']

    def is_free_msg(self) -> str:
        if self.__info['status'] == ChunkState.USED:
            return Color.colorify("Used", "red bold")
        else:
            return Color.colorify("Free", "green bold")

    def __str__(self) -> str:
        size = Color.colorify('Small' if self.is_small() else 'Large', 'bold')
        return (f"{Color.colorify('Chunk', 'yellow bold underline')}(addr={self.base_addr:#x}, "
                f"size={self.size:#x} ({size}), status={self.is_free_msg()})")

    def quarantine_msg(self) -> str:
        status = Color.colorify('Yes', 'green') if self.__info['status'] == ChunkState.FREE_QUARANTINE else Color.colorify('No', 'red')
        return f'In quarantine: {status}'

    def tcache_msg(self) -> str:
        status = Color.colorify('Yes', 'green') if self.__info['status'] == ChunkState.FREE_TCACHE else Color.colorify('No', 'red')
        return f'In tcache: {status}'

    def extent_info(self) -> str:
        msg = []
        extent = self.__info['extent']
        msg.append(f"Extent base: 0x{int(extent['e_addr']):x}")
        msg.append(f"Extent size: 0x{int(extent['e_bsize']):x}")
        if self.is_small():
            msg.append(f"Region index: 0x{self.__info['region_idx']:x}")
        return "\n".join(msg)

    def psprint(self) -> str:
        msg = []
        msg.append(str(self))
        if self.__info['status'] == ChunkState.USED:
            pass
        else:
            if quarantine.has_quarantine:
                msg.append(self.quarantine_msg())
            msg.append(self.tcache_msg())
        msg.append(self.extent_info())
        # else:
        #     msg.append(self.str_as_free())
        return "\n".join(msg) + "\n"

    @property
    def size(self) -> int:
        return self.__info['size']

    @property
    def szind(self) -> int:
        return self.__info['szind']

    @property
    def extent(self) -> int:
        return self.__info['extent']

    @property
    def base_addr(self) -> int:
        return self.__info['base_addr']

    @property
    def region_idx(self):
        return self.__info['region_idx']
     
    def is_small(self) -> bool:
        return self.size <= jemalloc.small_max


@register
class JemallocHeapCommand(GenericCommand):
    """Base command to get information about the Jemalloc heap structure."""

    _cmdline_ = "jheap"
    _syntax_  = f"{_cmdline_} (chunk|uaf|chunks)"

    def __init__(self) -> None:
        super().__init__(prefix=True)
        return

    @only_if_gdb_running
    def do_invoke(self, _: List[str]) -> None:
        self.usage()
        return


@register
class JemallocHeapChunkCommand(GenericCommand):
    """Display information on a Jemalloc heap chunk."""

    _cmdline_ = "jheap chunk"
    _syntax_  = f"{_cmdline_} [-h] address"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return

    @parse_arguments({"address": ""}, {})
    @only_if_gdb_running
    def do_invoke(self, _: List[str], **kwargs: Any) -> None:
        args = kwargs["arguments"]
        if not args.address:
            err("Missing chunk address")
            self.usage()
            return

        addr = parse_address(args.address)
        try:
            current_chunk = JemallocChunk(addr)
            gef_print(current_chunk.psprint())
        except:
            err("Invalid chunk")
        return


@register
class JemallocUAFCommand(GenericCommand, ScanCapsBase):
    """Scan memory for valid capabilities pointing to freed heap chunks.
    When scanning heap memory, only display capabilities that are stored in
    a heap chunk marked as `USED`. Heap block containing metadata is not
    scanned."""

    _cmdline_ = "jheap uaf"
    _syntax_  = f"{_cmdline_} [-h] [noheap]"

    def __init__(self) -> None:
        GenericCommand.__init__(self, complete=gdb.COMPLETE_NONE)
        ScanCapsBase.__init__(self, should_print = self.should_print_chunk)
        return

    # chunk_addr -> Address of heap chunk
    # srcloc -> Address where heap chunk cap was found
    # if we can't find a valid chunk at chunk_addr, it might be because
    # it is > 0x1000 bytes from the actual heap chunk's base
    # In this case, we use its source at srcloc to retrieve the full cap
    # and find the actual base as a second option
    def chunk_obj(self, chunk_addr, srcloc = None):
        idx = bisect.bisect_right(self.chunks.bases, chunk_addr)
        if idx == 0 or self.chunks.sizes[idx - 1] <= chunk_addr - self.chunks.bases[idx - 1]:
            chunkobj = None
            try:
                chunkobj = JemallocChunk(chunk_addr)
            except:
                if srcloc:
                    u = cached_lookup_type('__uintcap_t').pointer()
                    chunk_cap = gdb.Value(srcloc).cast(u).dereference()
                    _, _, start, _, _ = parse_cap_internal(chunk_cap)
                    try:
                        chunkobj = JemallocChunk(start)
                    except:
                        pass
            if chunkobj:
                self.chunks.bases.insert(idx, chunkobj.base_addr)
                self.chunks.sizes.insert(idx, chunkobj.size)
                self.chunks.objs.insert(idx, chunkobj)

        idx = bisect.bisect_right(self.chunks.bases, chunk_addr)
        if idx == 0 or self.chunks.sizes[idx - 1] <= chunk_addr - self.chunks.bases[idx - 1]:
            # warn(f"Chunk err @ {chunk_addr:x}")
            self.chunks.bases.insert(idx, chunk_addr)
            self.chunks.sizes.insert(idx, 1)
            self.chunks.objs.insert(idx, None) # Too bad we couldn't do anything

        idx = max(idx - 1, 0)
        return self.chunks.objs[idx]

    def should_print_chunk(self, scan_result):
        srcloc, chunk_addr = scan_result
        if self.heap.page_start <= srcloc < self.heap.page_end:
            chunk = self.chunk_obj(srcloc)
            if not chunk or \
                chunk.get_chunk_status() != ChunkState.USED:
                return False

        chunk = self.chunk_obj(chunk_addr, srcloc)
        if chunk and \
                chunk.get_chunk_status() != ChunkState.USED:
            return True
        else:
            return False

    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        # self.chunks.bases instead of self.chunks['base']
        class Chunks(object):
            def __init__(self):
                self.bases = [] # Base of heap chunk
                self.sizes = [] # Size of heap chunk
                self.objs = []  # JemallocChunk object representing chunk

        self.chunks = Chunks()
        self.heap = None
        for section in gef.memory.maps:
            if section.path == "[heap]":
                self.heap = section
                break
        if not self.heap:
            err("No heap region")
            return

        noheap = False # Don't display pointers from heap to heap
        if len(argv) == 1 and argv[0] == "noheap":
            noheap = True
        src = [
            (section.page_start, section.page_end)
            for section in self.get_readable_sections()
            if section.path != "[heap block]"
            and (not noheap or section.path != "[heap]")
        ]
        self.scan_memory(src, [(self.heap.page_start, self.heap.page_end)])

@register
class JemallocChunksCommand(GenericCommand):
    """Show all the chunks in USED state."""

    _cmdline_ = "jheap chunks"
    _syntax_  = f"{_cmdline_} [-h]"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        self.peek_nb_byte = 16
        return

    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        in_use = []
        for section in gef.memory.maps:
            if section.path == '[heap]':
                curr = section.page_start
                while curr < section.page_end:
                    chunk = JemallocChunk(curr)
                    extent_sz = int(chunk.extent['e_bsize'])
                    if chunk.is_small():
                        # Chunk is small, so a region in a slab
                        # We now have to check the bitmap of the slab
                        bitmap_idx = -1
                        idx = 0
                        bitmap_val = 0
                        chunk_sz = chunk.size
                        region_cnt = extent_sz / chunk_sz
                        # If there's a 0 in the position in the bitmap,
                        # the chunk may not really be in use as it can be inside the tcache
                        # We need to double check
                        while idx < region_cnt:
                            if idx % 64 == 0:
                                bitmap_idx += 1
                                bitmap_val = int(chunk.extent['e_slab_data']['bitmap'][bitmap_idx])
                            if bitmap_val & (1 << (idx % 64)) == 0:
                                region_chunk = JemallocChunk(curr + idx * chunk_sz)
                                if region_chunk.get_chunk_status() == ChunkState.USED:
                                    in_use.append(region_chunk)
                            idx += 1
                    else:
                        if chunk.get_chunk_status() == ChunkState.USED:
                            in_use.append(chunk)
                    curr += extent_sz
        for chunk in in_use:
            print(chunk)
            base = chunk.base_addr
            print(f"    [{hexdump(gef.memory.read(base, self.peek_nb_byte), self.peek_nb_byte, base=base, align = 18)}]")