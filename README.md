# GEF plugins

## Setup

Prerequisite: [gef with CHERI support](https://github.com/CTSRD-CHERI/gef)

```sh
# assuming this directory is named gef-plugins
echo gef config gef.extra_plugins_dir /path/to/gef-plugins >> ~/.gdbinit
```

## Additions

- Jemalloc heap manager: identify the heap memory regions, visualize heap allocations with metadata. Note that it is only tested on [CheriBSD jemalloc](https://github.com/CTSRD-CHERI/cheribsd/tree/main/contrib/jemalloc). 
    - `jheap chunk <address>`: inspect a heap allocation
    - `jheap chunks`: list in use heap allocations
    - `jheap uaf [noheap]`: scan for freed heap allocations that are pointed by valid capabilities in memory. Optionally, exclude capabilities stored in the heap.

- Snmalloc heap manager: tested on https://github.com/microsoft/snmalloc
    - `snheap localcache`: list entries in local cache `LocalCache` (also called small fast free lists)
    - `snheap slabs`: lists slabs in the core allocator (there can be multiple slabs per small size class, and large slabs)
    - `snheap remote`: lists the remote deallocation queue of the current of given thread(s)
    - `snheap freelists`: list entries in local cache `LocalCache` in the local allocator, the deallocation queue in remote allocators and active slab free lists in the core allocator
    - `snheap chunk <address>`: lists details about the `Alloc` and its slab. If the metaentry has the `REMOTE_BACKEND_MARKER` bit asserted, that is, the chunk is owned by the backend (not `Alloc`-bounded), then indicate it as a `Chunk`. Because backend chunks' metaentry are parsed differently depending on the specific `Range`, we can make a best guess of the owning `Range`. In the case that CHERI revocation is enabled, also print whether it is quarantined and its revocation bit value.

- Quarantine heap manager:
    - `mrs info`: display general information about the mrs quarantine, global state and the revocation bitmap.
    - `mrs chunk <address>`: query whether this chunk is owned by the allocator or quarantined. Also show shadow bitmap offset and value. The information we can query is limited because the capability load generation counter registers are not available to gdb in ring 3, so we can't inspect the kernel internal state of caprevoke unless debugging the kernel or using qemu.
    - `mrs quarantine`: print the quarantined chunks (and their shadow bit values of the allocation first word). 


## TODO

- More Jemalloc heap commands, like `arena(s)`, `extent(s)`, `tcache`, `slab(s)`
- PartitionAlloc
- Snmalloc:
    - Identify heap mappings in `vmmap` output (test with snmalloc as libc malloc)
    - `snheap chunks`: list allocated `Chunk`s by parsing the metaentries in pagemap that correspond to chunks returned by the backend. The pipe of Ranges return `Arena`-bounded pointers, and the backend casts object allocation pointers to `Chunk`-bounded pointers and metadata pointers to the corresponding `SlabMetadata` pointer. 
    - `snheap smallbuddy`: display world view of the chunks owned by the small buddy range in the backend
    - `snheap largebuddy`: display world view of the chunks owned by the large buddy range in the backend
    