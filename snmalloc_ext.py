
# look for SNMALLOC_REQUIRE_CONSTINIT for variables with static or thread-local storage duration
# I am querying gdb for finding struct addresses for stability during snmalloc struct refactoring
# Assumptions: snmalloc library with symbols (standard build)

# TODO: refactor constants
# TODO: account for customized configurations and PALs. Now it only works for Linux and the default config
# TODO: exception handling (extremely lacking)
# TODO: test on morello
# TODO: test as a shim and as the main allocator

from __future__ import annotations
from enum import Enum

def from_exp_mant(m_e: int):
    if GefSnmallocHeapManager.INTERMEDIATE_BITS > 0:
        m_e = m_e + 1
        mask = (1 << GefSnmallocHeapManager.INTERMEDIATE_BITS) - 1
        m = m_e & mask
        e = m_e >> GefSnmallocHeapManager.INTERMEDIATE_BITS
        b = 0 if e == 0 else 1
        shifted_e = e - b
        extended_m = (m + (b << GefSnmallocHeapManager.INTERMEDIATE_BITS))
        return extended_m << (shifted_e + GefSnmallocHeapManager.MIN_ALLOC_BITS)
    else:
        return 1 << (m_e + GefSnmallocHeapManager.MIN_ALLOC_BITS)


class GefSnmallocHeapManager(GefManager):
    """Snmalloc heap manager."""
    # configurable constants (TODO: read from gdb)
    INTERMEDIATE_BITS = 2 # default
    MIN_ALLOC_BITS = 4 # default
    REMOTE_WITH_BACKEND_MARKER_ALIGN = 1 << 7
    TAG = 0x40  # TODO: read from _ZN8snmalloc11sizeclass_t3TAGE
    GRANULARITY_BIT = 14
    METAENTRY_SIZE = 0x10

    def __init__(self) -> None:
        self.reset_caches()
        return
    
    def reset_caches(self) -> None:
        super().reset_caches()
        self.NUM_SMALL_SIZECLASSES = None
        self.__pagemap_body = None
        return
    
    @property
    def num_small_sizeclasses(self) -> Optional[int]:
        if not self.NUM_SMALL_SIZECLASSES:
            try:
                self.NUM_SMALL_SIZECLASSES = gdb.parse_and_eval("'snmalloc::NUM_SMALL_SIZECLASSES'")
            except:
                pass
        return self.NUM_SMALL_SIZECLASSES
    
    @property
    def pagemap_body(self) -> Optional[int]:
        if not self.__pagemap_body:
            try:
                self.__pagemap_body = GefSnmallocHeapManager.find_pagemap_body()
            except:
                pass
        return self.__pagemap_body
    
    @staticmethod
    def find_pagemap_body() -> int:
        """Find the Pagemap body."""
        # TODO: coupled with Linux!
        pagemap_body = parse_address("_ZN8snmalloc12BasicPagemapINS_8PALLinuxENS_11FlatPagemapILm14ENS_20DefaultPagemapEntryTINS_19DefaultSlabMetadataEEES1_Lb0EEES5_Lb0EE15concretePagemapE->body")
        return pagemap_body
    
    @staticmethod
    def find_localalloc_addr() -> int:
        """Find the address of the current thread's LocalAlloc."""
        try:
            localalloc_addr = parse_address("&_ZZN8snmalloc11ThreadAlloc3getEvE5alloc")
        except gdb.error:
            # In binaries not linked with pthread...
            print("UNIMPLEMENTED")
            pass
        return localalloc_addr
    
    @staticmethod
    def decode_ras(ras: int) -> Tuple[int, int]:
        sizeclass = ras & (GefSnmallocHeapManager.REMOTE_WITH_BACKEND_MARKER_ALIGN - 1)
        # remote = bits::align_down(ras, MetaEntry.REMOTE_WITH_BACKEND_MARKER_ALIGN)
        remote = ras & ~(GefSnmallocHeapManager.REMOTE_WITH_BACKEND_MARKER_ALIGN - 1)
        return remote, sizeclass
    
    # TODO: INCOMPLETE, see sizeclasstable.h
    @staticmethod
    def from_small_class(sizeclass: int) -> int:
        """Convert from small sizeclass encoding to sizeclass encoding."""
        return GefSnmallocHeapManager.TAG | sizeclass
    
    @staticmethod
    def from_large_class(sizeclass: int) -> int:
        """Convert from large sizeclass encoding to sizeclass encoding."""
        return sizeclass
    
    @staticmethod
    def as_small(sizeclass: int) -> int:
        """Convert from sizeclass encoding to small sizeclass encoding."""
        return sizeclass & (GefSnmallocHeapManager.TAG - 1)
    
    @staticmethod
    def as_large(sizeclass: int) -> int:
        """Convert from sizeclass encoding to large sizeclass encoding."""
        return 64 - (sizeclass & (GefSnmallocHeapManager.TAG - 1))
    
    @staticmethod
    def sizeclass_to_size(sizeclass: int) -> int:
        """Convert from small sizeclass encoding to size."""
        # fast small
        # bits::from_exp_mant<INTERMEDIATE_BITS, MIN_ALLOC_BITS>(sizeclass);
        return from_exp_mant(sizeclass)
    
    @staticmethod
    def sizeclass_full_to_size(sizeclass: int) -> int:
        """Convert from sizeclass encoding to size."""
        assert(False)
        return
    
    @staticmethod
    def sizeclass_full_to_slab_size(sizeclass: int) -> int:
        """Convert from sizeclass encoding to slab size."""
        assert(False)
    
    @staticmethod
    def sizeclass_to_slab_size(sizeclass: int) -> int:
        """Convert from small sizeclass encoding to slab size."""
        assert(False)

# initialize snmalloc heap manager
snmallocheap = GefSnmallocHeapManager()

## ALLOCATION
    
class SnmallocChunkType(Enum):
    ALLOC = 0  # owned by the frontend
    CHUNK = 1  # owned by the backend


class SnmallocChunkBase:
    """Base class for chunks."""
    def __init__(self, address: int):
        self.address = address
        self.reset()

    def reset(self):
        self.__metaentry = None
        return
    
    @property
    def metaentry(self) -> MetaEntry:
        if not self.__metaentry:
            # metaentry at pagemap.body + (p >> GRANULARITY_BIT) * METAENTRY_SIZE
            idx = self.address >> GefSnmallocHeapManager.GRANULARITY_BIT
            metaentry_addr = snmallocheap.pagemap_body + idx * MetaEntry.sizeof()
            self.__metaentry = MetaEntry(metaentry_addr)
        return self.__metaentry


class SnmallocAlloc(SnmallocChunkBase):
    """Alloc structure owned by the frontend."""
    def __init__(self, addr: int):
        super().__init__(addr)
        self.chunk_type = SnmallocChunkType.ALLOC
        self.reset()

    def reset(self):
        super().reset()
        self.__slabmeta = None
        self.__sc = None
        self.__size = None
        return
    
    @property
    def slabmeta(self) -> SlabMetadata:
        if not self.__slabmeta:
            self.__slabmeta = SlabMetadata(self.metaentry.meta)
        return self.__slabmeta
    
    @property
    def sc(self) -> int:
        if not self.__sc:
            self.__sc = GefSnmallocHeapManager.as_large(self.metaentry.sizeclass) if self.slabmeta.large else GefSnmallocHeapManager.as_small(self.metaentry.sizeclass)
        return self.__sc
    
    @property
    def size(self) -> int:
        if not self.__size:
            if self.slabmeta.large:
                self.__size = 1 << self.sc
            else:
                self.__size = GefSnmallocHeapManager.sizeclass_to_size(self.sc)
        return self.__size
    
    def __str__(self) -> str:
        return (f"{Color.colorify('Alloc', 'yellow bold underline')}(addr={self.address:#x}, "
                f"large={self.slabmeta.large}, "
                f"sc={self.sc:d}, size={self.size:#x}, "
                f"remote={self.metaentry.remote:#x}, meta={self.metaentry.meta:#x})")


class SnmallocSmallBuddyChunk(SnmallocChunkBase):
    pass


class SnmallocLargeBuddyChunk(SnmallocChunkBase):
    pass

## ITERATORS

class SeqSetNode:
    """A node in a sequential set (used to group metadata)."""
    def __init__(self, address: int):
        self.address = address
        self.reset()

    def reset(self):
        self.__next = None
        self.__prev = None

    @property
    def next(self) -> int:
        if not self.__next:
            try:
                self.__next = int(dereference(self.address))
            except gdb.error:
                print("UNIMPLEMENTED")
                pass
        return self.__next

    @property
    def prev(self) -> int:
        if not self.__prev:
            try:
                self.__prev = int(dereference(self.address + gef.arch.ptrsize))
            except gdb.error:
                print("UNIMPLEMENTED")
                pass
        return self.__prev
    
    @staticmethod
    def sizeof() -> int:
        return 0x10


class SeqSet:
    """Doubly-linked cyclic list linked using T::node field."""
    def __init__(self, addr: int):
        self.address = addr
        self.reset()

    def reset(self):
        self.__head = None

    @property
    def head(self) -> SeqSetNode:
        if not self.__head:
            self.__head = SeqSetNode(self.address)
        return self.__head
    
    def get_objects(self) -> Tuple[List[int], int]:
        """Return count -1 if loop is detected."""
        ret = []
        visited = set()
        curr_node = self.head
        while curr_node.next != self.head.address:
            ret.append(curr_node.next)
            visited.add(curr_node.next)
            if len(visited) != len(ret):
                return ret, -1
            curr_node = SeqSetNode(curr_node.next)
        return ret, len(ret)


class SnmallocFreelistObject(SnmallocAlloc):
    """Freelist object. (Signed) prev field is present depending on configured security mitigations."""
    def __init__(self, addr: int):
        super().__init__(addr)
        self.reset()

    def reset(self):
        super().reset()
        self.__next_object = None
        self.__prev_encoded = None 

    @property
    def next_object(self) -> int:
        if not self.__next_object:
            self.__next_object = int(dereference(self.address))
        return self.__next_object
    
    @property
    def prev_encoded(self) -> int:
        if not self.__prev_encoded:
            self.__prev_encoded = int(dereference(self.address + gef.arch.ptrsize))
        return self.__prev_encoded


class SnmallocFreelistIter:
    """Used to iterate a freelist in object space. It is null terminated."""
    def __init__(self, address: int):
        self.address = address
        self.reset()

    def reset(self):
        self.__curr = None

    @property
    def curr(self) -> SnmallocFreelistObject:
        if not self.__curr:
            self.__curr = SnmallocFreelistObject(self.address)
        return self.__curr
    
    def get_objects(self) -> Tuple[List[int], int]:
        """Return count -1 if loop is detected."""
        curr_node = self.curr
        ret = []
        visited = set()
        while curr_node.next_object != 0:
            ret.append(curr_node.next_object)
            visited.add(curr_node.next_object)
            if len(visited) != len(ret):
                return ret, -1
            curr_node = SnmallocFreelistObject(curr_node.next_object)
        return ret, len(ret)


class SnmallocFreelistBuilder:
    """Used to build a freelist in object space. Hence, end is a pointer to the last FreelistObject.
    A builder can contain two freelist queues depending on whether `random_preserve` mitigation is enabled.
    """
    def __init__(self, addr: int, random_builder: bool = False):
        self.address = addr
        self.LENGTH = 1 if not random_builder else 2
        self.reset()

    def reset(self):
        self.__head = None
        self.__end = None
        self.__length = None

    @property
    def head(self) -> Tuple[SnmallocFreelistObject, Optional[SnmallocFreelistObject]]:
        if not self.__head:
            self.__head = SnmallocFreelistObject(self.address)
        return self.__head, None # TODO: random builder security mitigation
    
    @property
    def end(self) -> Tuple[SnmallocFreelistObject, Optional[SnmallocFreelistObject]]:
        if not self.__end:
            self.__end = SnmallocFreelistObject(self.address + gef.arch.ptrsize)
        return self.__end, None # TODO: random builder security mitigation
    
    @property
    def length(self) -> Optional[Tuple[int, int]]:
        return None # TODO: random builder security mitigation
    
    def get_objects(self, i: int = 0) -> Tuple[List[int], int]:
        """Refer to the freelist i (0 or 1). Return count -1 if loop is detected."""
        if self.address == 0: return [], 0
        head_address = self.head[0].address # TODO: random builder security mitigation
        # empty case
        if head_address == self.end[0].next_object:
            return [], 0
        ret = []
        visited = set()
        curr_node = self.head[0]
        while curr_node.address != 0:
            ret.append(curr_node.next_object)
            visited.add(curr_node.next_object)
            if len(ret) != len(visited):
                return ret, -1
            if curr_node.next_object == self.end[0].next_object:
                break
            curr_node = SnmallocFreelistObject(curr_node.next_object)
        return ret, len(ret)

## METADATA

class SlabMetadata:
    """Frontend slab metadata."""
    def __init__(self, addr: int):
        self.address = addr
        self.reset()

    def reset(self):
        self.__node = None
        self.__free_queue = None
        self.__needed = None
        self.__sleeping = None
        self.__large = None

    @property
    def node(self) -> SeqSetNode:
        if not self.__node:
            self.__node = SeqSetNode(self.address)
        return self.__node
    
    @property
    def free_queue(self): # -> SnmallocFreelistBuilder, Python typing cannot declare circular dependencies
        if not self.__free_queue:
            self.__free_queue = SnmallocFreelistBuilder(self.address + SeqSetNode.sizeof())
        return self.__free_queue
    
    @property
    def needed(self) -> int:
        if not self.__needed:
            self.__needed = int(gdb.parse_and_eval(f"(*('snmalloc::DefaultSlabMetadata'*){self.address})->needed_"))
        return self.__needed
    
    @property
    def sleeping(self) -> bool:
        if not self.__sleeping:
            self.__sleeping = int(gdb.parse_and_eval(f"(*('snmalloc::DefaultSlabMetadata'*){self.address})->sleeping_")) != 0
        return self.__sleeping
    
    @property
    def large(self) -> bool:
        if not self.__large:
            self.__large = int(gdb.parse_and_eval(f"(*('snmalloc::DefaultSlabMetadata'*){self.address})->large_")) != 0
        return self.__large
    
    def __str__(self) -> str:
        return f"SlabMetadata[addr={self.address:#x}, needed={self.needed:d}, sleeping={self.sleeping}, large={self.large}]"


class MetaEntry:
    """Entry in the metadata pagemap. Both chunks owned by the frontend and the backend have an associated metaentry."""
    def __init__(self, address):
        self.address = address
        self.reset()

    def reset(self):
        self.__ras = None
        self.__meta = None
        return

    @property
    def ras(self) -> int:
        if not self.__ras:
            self.__ras = u64(gef.memory.read(self.address + gef.arch.ptrsize, gef.arch.ptrsize)) # TODO: hardcoded pointer size, doesn't work on CHERI
        return self.__ras
    
    @property
    def meta(self) -> int:
        if not self.__meta:
            self.__meta = u64(gef.memory.read(self.address, gef.arch.ptrsize))
        return self.__meta
    
    @property
    def remote(self) -> int:
        remote, _ = GefSnmallocHeapManager.decode_ras(self.ras)
        return remote
    
    @property
    def sizeclass(self) -> int:
        _, sizeclass = GefSnmallocHeapManager.decode_ras(self.ras)
        return sizeclass
    
    @staticmethod
    def sizeof() -> int:
        return 2 * gef.arch.ptrsize

## ALLOCATORS

class RemoteAllocator:
    """Remote deallocation queue."""
    def __init__(self, addr: int):
        self.address = addr
        self.reset()

    def reset(self):
        self.__back = None
        self.__front = None

    @property
    def back(self) -> SnmallocFreelistObject:
        if not self.__back:
            back_addr = parse_address(f"&(*('snmalloc::RemoteAllocator' *){self.address})->back")
            self.__back = SnmallocFreelistObject(back_addr)
        return self.__back
    
    @property
    def front(self) -> SnmallocFreelistObject:
        if not self.__front:
            front_addr = parse_address(f"&(*('snmalloc::RemoteAllocator' *){self.address})->front")
            self.__front = SnmallocFreelistObject(front_addr)
        return self.__front
    
    def get_objects(self):
        # empty case, even though there is a fake stub
        if self.back.next_object == 0:
            return [], 0
        ret = []
        visited = set()
        curr_node = self.front
        while curr_node.next_object != self.back.next_object and curr_node.next_object != 0:
            ret.append(curr_node.next_object)
            visited.add(curr_node.next_object)
            if len(ret) != len(visited):
                return ret, -1
            curr_node = SnmallocFreelistObject(curr_node.next_object)
        return ret, len(ret)
    
    def __str__(self) -> str:
        return f"RemoteAllocator(addr={self.address:#x})"


class CoreAlloc:
    """Core allocator structure."""

    class SlabMetadataCache:
        """Wrapper around a SeqSet of slab metadata."""
        def __init__(self, head: SeqSet, unused: int, length: int):
            self.available = head # sequence set head
            self.unused = unused
            self.length = length
        
        @staticmethod
        def sizeof() -> int:
            return 0x18

    def __init__(self, addr: int):
        self.address = addr
        self.reset()

    def reset(self):
        self.__alloc_classes = None
        self.__laden = None
        self.__remote_alloc = None

    @property
    def alloc_classes(self) -> List[SlabMetadataCache]:
        if not self.__alloc_classes:
            try:
                self.__alloc_classes = [None for _ in range(snmallocheap.num_small_sizeclasses)]
                array_base = parse_address(f"&(*('snmalloc::CoreAllocator<snmalloc::StandardConfig>'*){self.address:#x})->alloc_classes")
                for i in range(snmallocheap.num_small_sizeclasses):
                    cache_struct_base = array_base + i * self.SlabMetadataCache.sizeof()
                    head = SeqSet(cache_struct_base)
                    unused = u16(gef.memory.read(cache_struct_base + 0x10, 2))
                    length = u16(gef.memory.read(cache_struct_base + 0x10 + 0x4, 2))
                    cache_struct = self.SlabMetadataCache(head, unused, length)
                    self.__alloc_classes[i] = cache_struct
            except gdb.error:
                print("UNIMPLEMENTED")
                pass
        return self.__alloc_classes
    
    @property
    def laden(self) -> SeqSet:
        if not self.__laden:
            try:
                laden_base = parse_address(f"&(*('snmalloc::CoreAllocator<snmalloc::StandardConfig>'*){self.address:#x})->laden")
                self.__laden = SeqSet(laden_base)
            except gdb.error:
                print("UNIMPLEMENTED")
                pass
        return self.__laden

    def __str__(self) -> str:
        return (f"{Color.colorify('CoreAlloc', 'red bold underline')}(addr={self.address:#x})")
    
    @property
    def remote_alloc(self):
        # TODO: assuming queue is inline, so RemoteAllocator and not RemoteAllocator*
        if not self.__remote_alloc:
            self.__remote_alloc = RemoteAllocator(parse_address(f"&(*('snmalloc::CoreAllocator<snmalloc::StandardConfig>'*){self.address:#x})->remote_alloc"))
        return self.__remote_alloc


class LocalAlloc:
    """Local allocator structure."""
    def __init__(self, addr: int):
        self.address = addr
        self.reset()

    def reset(self):
        self.__small_fast_free_lists = None
        self.__corealloc = None

    @property
    def small_fast_free_lists(self) -> List[SnmallocFreelistIter]:
        """The list of HeadPtr of the small fast freelists."""
        if not self.__small_fast_free_lists:
            localcache_addr = self.find_localcache_addr()
            lists = []
            for i in range(snmallocheap.num_small_sizeclasses):
                head_ptr = SnmallocFreelistIter(localcache_addr + i*gef.arch.ptrsize)
                lists.append(head_ptr)
            self.__small_fast_free_lists = lists
        return self.__small_fast_free_lists
    
    @property
    def corealloc(self) -> CoreAlloc:
        """The core allocator that this local allocator owns."""
        if not self.__corealloc:
            self.__corealloc = CoreAlloc(self.find_corealloc_addr())
        return self.__corealloc
    
    def find_localcache_addr(self) -> Optional[int]:
        """Find the address of the current thread's LocalAlloc local cache."""
        try:
            localcache_addr = parse_address(f"&(*('snmalloc::Alloc' *){hex(self.address)})->local_cache->small_fast_free_lists")
        except gdb.error:
            # In binaries not linked with pthread...
            print("UNIMPLEMENTED")
            pass
        return localcache_addr

    def find_corealloc_addr(self) -> Optional[int]:
        try:
            corealloc_addr = parse_address(f"(*('snmalloc::Alloc' *){hex(self.address)})->core_alloc")
        except gdb.error:
            # In binaries not linked with pthread
            print("UNIMPLEMENTED")
            pass
        return corealloc_addr
    
    def __str__(self) -> str:
        return (f"{Color.colorify('LocalAlloc', 'red bold underline')}(addr={self.address:#x}, "
                f"corealloc={self.corealloc.address:#x})")


## COMMANDS
    
def print_localcache(thread, localalloc: LocalAlloc) -> None:
    gef_print(titlify(f"Thread {thread.num:d}: localcache for LocalAlloc(addr={localalloc.address:#x})"))
    
    # loop through all the free lists
    for i in range(snmallocheap.num_small_sizeclasses):
        freelist_iter = localalloc.small_fast_free_lists[i]
        allocs, count = freelist_iter.get_objects()
        if count == 0:
            continue
        msg, limit = [], 5
        for alloc in allocs[:limit-1]:
            chunk = SnmallocAlloc(alloc)
            msg.append(f"{RIGHT_ARROW} {chunk!s} ")
        if len(allocs) > limit - 1:
            chunk = SnmallocAlloc(allocs[-1])
            msg.append(f"{RIGHT_ARROW} ... {RIGHT_ARROW} {chunk!s} ")
        if count == -1:
            msg.append("[Loop detected]")
        gef_print(f"small_fast_free_lists[idx={i:d}, count={count}] ", end="")
        gef_print("".join(msg))
        gef_print("")
    return


def print_alloc_classes(thread, localalloc: LocalAlloc) -> None:
    corealloc = localalloc.corealloc

    # parse active slabs
    gef_print(titlify(f"Thread {thread.num:d}: active slabs for CoreAlloc(addr={corealloc.address:#x}) {LEFT_ARROW} LocalAlloc(addr={localalloc.address:#x})"))
    no_active_slabs = True
    for i in range(snmallocheap.num_small_sizeclasses):
        slabmetacache = corealloc.alloc_classes[i]
        allocs, count = slabmetacache.available.get_objects()
        if count == 0:
            continue
        no_active_slabs = False
        msg, limit = [], 5
        for alloc in allocs[:limit-1]:
            chunk = SlabMetadata(alloc)
            msg.append(f"{RIGHT_ARROW} {chunk!s} ")
        if len(allocs) > limit - 1:
            chunk = SlabMetadata(allocs[-1])
            msg.append(f"{RIGHT_ARROW} ... {RIGHT_ARROW} {chunk!s} ")
        if count == -1:
            msg.append("[Loop detected]")
        gef_print(f"SlabMetadataCache[idx={i:d}, unused={slabmetacache.unused:d}, length={slabmetacache.length:d}] ", end="")
        gef_print("".join(msg))
        gef_print("")
    if no_active_slabs:
        info("No active slabs")
    return


def print_laden(thread, localalloc: LocalAlloc) -> None:
    corealloc = localalloc.corealloc

    # parse laden
    gef_print(titlify(f"Thread {thread.num:d}: laden for CoreAlloc(addr={corealloc.address:#x}) {LEFT_ARROW} LocalAlloc(addr={localalloc.address:#x})"))
    large_or_inactive_slabs, count = corealloc.laden.get_objects()
    if count == 0:
        info("No laden slabs")
    else:
        for node in large_or_inactive_slabs:
            msg, limit = [], 5
            slabmetadata = SlabMetadata(node)
            objects, count = slabmetadata.free_queue.get_objects()
            for object in objects[:limit-1]:
                chunk = SnmallocAlloc(object)
                msg.append(f"{RIGHT_ARROW} {chunk!s} ")
            if count > limit - 1:
                chunk = SnmallocAlloc(objects[-1])
                msg.append(f"{RIGHT_ARROW} ... {RIGHT_ARROW} {chunk!s} ")
            if count == -1:
                msg.append("[Loop detected]")
            gef_print(f"{slabmetadata!s} ")
            gef_print("".join(msg))
            if count != 0:
                gef_print("")
    return


def print_remote_freelist(thread, localalloc: LocalAlloc) -> None:
    corealloc = localalloc.corealloc
    remotealloc = corealloc.remote_alloc

    # parse remote deallocation list
    gef_print(titlify(f"Thread {thread.num:d}: remote deallocation queue for RemoteAllocator(addr={remotealloc.address:#x}) {LEFT_ARROW} CoreAlloc(addr={corealloc.address:#x})"))
    objects, count = remotealloc.get_objects()
    if count == 0:
        info("Remote deallocation list is empty")
        return
    msg, limit = [], 5
    for object in objects[:limit-1]:
        msg.append(f"{RIGHT_ARROW} {object!s} ")
    if count > limit - 1:
        msg.append(f"{RIGHT_ARROW} ... {RIGHT_ARROW} {objects[-1]!s} ")
    if count == -1:
        msg.append("[Loop detected]")
    gef_print("".join(msg))
    return


@register
class SnmallocHeapCommand(GenericCommand):
    """Base command to get information about the snmalloc heap structure."""

    _cmdline_ = "snheap"
    _syntax_  = f"{_cmdline_} (localcache|slabs|remote|freelists|chunk)"

    def __init__(self) -> None:
        super().__init__(prefix=True)
        return

    @only_if_gdb_running
    def do_invoke(self, _: List[str]) -> None:
        self.usage()
        return


@register
class SnmallocHeapLocalcacheCommand(GenericCommand):
    """List entries in local cache (also called small fast free lists) of the local allocator of the given thread id(s)."""

    _cmdline_ = "snheap localcache"
    _syntax_  = f"{_cmdline_} [-h] [all] [thread_ids...]"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        current_thread = gdb.selected_thread()
        if current_thread is None:
            err("Couldn't find current thread")
            return

        # As a nicety, we want to display threads in ascending order by gdb number
        threads = sorted(gdb.selected_inferior().threads(), key=lambda t: t.num)
        if argv:
            if "all" in argv:
                tids = [t.num for t in threads]
            else:
                tids = self.check_thread_ids([int(a) for a in argv])
        else:
            tids = [current_thread.num]

        # loop through all selected threads (default current thread)
        for thread in threads:
            if thread.num not in tids:
                continue
            thread.switch()

            localalloc_addr = GefSnmallocHeapManager.find_localalloc_addr()
            if localalloc_addr == 0:
                info(f"Uninitialized LocalAlloc for thread {thread.num:d}")
                continue
            print_localcache(thread, LocalAlloc(localalloc_addr))

        current_thread.switch()
        return


@register
class SnmallocHeapSlabsCommand(GenericCommand):
    """List slabs in the core allocator owned by the local allocator of the given thread id(s)."""

    _cmdline_ = "snheap slabs"
    _syntax_  = f"{_cmdline_} [-h] [all] [thread_ids...]"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        current_thread = gdb.selected_thread()
        if current_thread is None:
            err("Couldn't find current thread")
            return

        # As a nicety, we want to display threads in ascending order by gdb number
        threads = sorted(gdb.selected_inferior().threads(), key=lambda t: t.num)
        if argv:
            if "all" in argv:
                tids = [t.num for t in threads]
            else:
                tids = self.check_thread_ids([int(a) for a in argv])
        else:
            tids = [current_thread.num]

        # loop through all selected threads (default current thread)
        for thread in threads:
            if thread.num not in tids:
                continue
            thread.switch()

            localalloc_addr = GefSnmallocHeapManager.find_localalloc_addr()
            if localalloc_addr == 0:
                info(f"Uninitialized LocalAlloc for thread {thread.num:d}")
                continue
            localalloc = LocalAlloc(localalloc_addr)
            print_alloc_classes(thread, localalloc)

            print_laden(thread, localalloc)

        current_thread.switch()


@register
class SnmallocHeapChunkCommand(GenericCommand):
    """List details about the slab that owns the given heap address and its free list."""

    _cmdline_ = "snheap chunk"
    _syntax_  = f"{_cmdline_} [-h] address"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @parse_arguments({"address": ""}, {})
    @only_if_gdb_running
    def do_invoke(self, _: List[str], **kwargs: Any) -> None:
        args = kwargs["arguments"]
        if not args.address:
            err("Missing heap address")
            self.usage()
            return

        addr = parse_address(args.address)
        # get the slabmetadata
        alloc = SnmallocAlloc(addr) # this works for any address under the same metaentry
        slabmetadata = alloc.slabmeta
        objects, count = slabmetadata.free_queue.get_objects()
        msg, limit = [], 5
        for object in objects[:limit-1]:
            chunk = SnmallocAlloc(object)
            msg.append(f"{RIGHT_ARROW} {chunk!s} ")
        if count > limit - 1:
            chunk = SnmallocAlloc(objects[-1])
            msg.append(f"{RIGHT_ARROW} ... {RIGHT_ARROW} {chunk!s} ")
        if count == -1:
            msg.append("[Loop detected]")
        gef_print(f"{alloc!s}")
        gef_print(f"{slabmetadata!s} ", end="")
        if count == 0:
            gef_print("")
            info("Free list is empty")
        else: gef_print("".join(msg))
        return


@register
class SnmallocHeapRemoteCommand(GenericCommand):
    """List the remote deallocation queue of the current of given thread(s)."""

    _cmdline_ = "snheap remote"
    _syntax_  = f"{_cmdline_} [-h] [all] [thread_ids...]"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        current_thread = gdb.selected_thread()
        if current_thread is None:
            err("Couldn't find current thread")
            return

        # As a nicety, we want to display threads in ascending order by gdb number
        threads = sorted(gdb.selected_inferior().threads(), key=lambda t: t.num)
        if argv:
            if "all" in argv:
                tids = [t.num for t in threads]
            else:
                tids = self.check_thread_ids([int(a) for a in argv])
        else:
            tids = [current_thread.num]

        # loop through all selected threads (default current thread)
        for thread in threads:
            if thread.num not in tids:
                continue
            thread.switch()

            localalloc_addr = GefSnmallocHeapManager.find_localalloc_addr()
            if localalloc_addr == 0:
                info(f"Uninitialized LocalAlloc for thread {thread.num:d}")
                continue
            localalloc = LocalAlloc(localalloc_addr)
            print_remote_freelist(thread, localalloc)

        current_thread.switch()
        return


@register
class SnmallocHeapFreelistsCommand(GenericCommand):
    """List entries in local cache, the deallocation queue in remote allocators and active slab free lists in the core allocator."""

    _cmdline_ = "snheap freelists"
    _syntax_  = f"{_cmdline_} [-h] [all] [thread_ids...]"

    def __init__(self) -> None:
        super().__init__(complete=gdb.COMPLETE_NONE)
        return
    
    @only_if_gdb_running
    def do_invoke(self, argv: List[str], **kwargs: Any) -> None:
        current_thread = gdb.selected_thread()
        if current_thread is None:
            err("Couldn't find current thread")
            return

        # As a nicety, we want to display threads in ascending order by gdb number
        threads = sorted(gdb.selected_inferior().threads(), key=lambda t: t.num)
        if argv:
            if "all" in argv:
                tids = [t.num for t in threads]
            else:
                tids = self.check_thread_ids([int(a) for a in argv])
        else:
            tids = [current_thread.num]

        # loop through all selected threads (default current thread)
        for thread in threads:
            if thread.num not in tids:
                continue
            thread.switch()
            localalloc_addr = GefSnmallocHeapManager.find_localalloc_addr()
            if localalloc_addr == 0:
                info(f"Uninitialized LocalAlloc for thread {thread.num:d}")
                continue
            localalloc = LocalAlloc(localalloc_addr)
            print_localcache(thread, localalloc)
            print_alloc_classes(thread, localalloc)
            print_laden(thread, localalloc)
            print_remote_freelist(thread, localalloc)

        current_thread.switch()
        return
